"""@bruin

name: ingestion.trips

type: python

image: python:3.11

connection: duckdb-default

materialization:
  type: table
  strategy: append

columns:
  - name: taxi_type
    type: string
    description: Taxi service type (e.g. yellow, green)
  - name: pickup_datetime
    type: timestamp
    description: Trip pickup timestamp
  - name: dropoff_datetime
    type: timestamp
    description: Trip dropoff timestamp
  - name: pickup_location_id
    type: integer
    description: TLC zone ID where the trip started
  - name: dropoff_location_id
    type: integer
    description: TLC zone ID where the trip ended
  - name: passenger_count
    type: integer
    description: Number of passengers on the trip
  - name: trip_distance
    type: float
    description: Trip distance in miles
  - name: fare_amount
    type: float
    description: Base fare amount in USD
  - name: total_amount
    type: float
    description: Total trip amount in USD
  - name: payment_type
    type: integer
    description: Payment type ID (joins to ingestion.payment_lookup)
  - name: extracted_at
    type: timestamp
    description: UTC timestamp when this record was ingested

@bruin"""

import io
import json
import os
from datetime import date
from typing import Dict, Iterable, List

import pandas as pd
import requests


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _month_starts(start: date, end: date) -> Iterable[date]:
    """
    Yield the first day of each month that overlaps the [start, end) window.
    """
    year = start.year
    month = start.month
    while True:
        current = date(year, month, 1)
        if current >= end:
            break
        yield current
        month += 1
        if month > 12:
            month = 1
            year += 1


def _load_parquet_from_url(url: str) -> pd.DataFrame:
    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code != 200:
        # For this learning project, log and skip missing months instead of failing the whole run.
        print(f"Skipping URL {url} (status={resp.status_code})")
        return pd.DataFrame()
    return pd.read_parquet(io.BytesIO(resp.content))


def _standardize_schema(df: pd.DataFrame, taxi_type: str) -> pd.DataFrame:
    """
    Normalize column names across yellow/green schemas while keeping raw content.
    """
    if df.empty:
        return df

    # Normalize to lowercase first for consistent renames.
    df = df.rename(columns={c: c.lower() for c in df.columns})

    rename_map: Dict[str, str] = {}
    if "tpep_pickup_datetime" in df.columns:
        rename_map["tpep_pickup_datetime"] = "pickup_datetime"
    if "lpep_pickup_datetime" in df.columns:
        rename_map["lpep_pickup_datetime"] = "pickup_datetime"
    if "tpep_dropoff_datetime" in df.columns:
        rename_map["tpep_dropoff_datetime"] = "dropoff_datetime"
    if "lpep_dropoff_datetime" in df.columns:
        rename_map["lpep_dropoff_datetime"] = "dropoff_datetime"
    if "pulocationid" in df.columns:
        rename_map["pulocationid"] = "pickup_location_id"
    if "dolocationid" in df.columns:
        rename_map["dolocationid"] = "dropoff_location_id"

    df = df.rename(columns=rename_map)

    # Ensure datetime types for pickup/dropoff.
    if "pickup_datetime" in df.columns:
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    if "dropoff_datetime" in df.columns:
        df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")

    # Add taxi_type column for downstream partitioning/analytics.
    df["taxi_type"] = taxi_type

    return df


def materialize() -> pd.DataFrame:
    """
    Ingest NYC taxi trip data from the TLC public parquet files.

    - Uses BRUIN_START_DATE / BRUIN_END_DATE to determine the time window.
    - Uses `taxi_types` from BRUIN_VARS to decide which taxi types to ingest.
    - Fetches monthly parquet files, standardizes a minimal schema, and returns a DataFrame.
    """
    start_date_str = os.environ.get("BRUIN_START_DATE")
    end_date_str = os.environ.get("BRUIN_END_DATE")
    if not start_date_str or not end_date_str:
        raise RuntimeError("BRUIN_START_DATE and BRUIN_END_DATE must be set for this asset.")

    start_date = _parse_date(start_date_str)
    end_date = _parse_date(end_date_str)

    vars_raw = os.environ.get("BRUIN_VARS", "{}")
    vars_json = json.loads(vars_raw)
    taxi_types: List[str] = vars_json.get("taxi_types", ["yellow"])

    frames: List[pd.DataFrame] = []

    for taxi_type in taxi_types:
        for month_start in _month_starts(start_date, end_date):
            url = f"{BASE_URL}/{taxi_type}_tripdata_{month_start.year}-{month_start.month:02d}.parquet"
            print(f"Fetching {url}")
            df = _load_parquet_from_url(url)
            if df.empty:
                continue

            df = _standardize_schema(df, taxi_type=taxi_type)

            # Filter to the requested window on pickup_datetime if present.
            if "pickup_datetime" in df.columns:
                mask = (df["pickup_datetime"] >= pd.Timestamp(start_date)) & (
                    df["pickup_datetime"] < pd.Timestamp(end_date)
                )
                df = df.loc[mask]

            if df.empty:
                continue

            df["extracted_at"] = pd.Timestamp.utcnow()

            frames.append(df)

    if not frames:
        # Return an empty DataFrame with the expected columns so materialization still succeeds.
        return pd.DataFrame(
            columns=[
                "taxi_type",
                "pickup_datetime",
                "dropoff_datetime",
                "pickup_location_id",
                "dropoff_location_id",
                "passenger_count",
                "trip_distance",
                "fare_amount",
                "total_amount",
                "payment_type",
                "extracted_at",
            ]
        )

    return pd.concat(frames, ignore_index=True) 

