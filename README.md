# GTC ML Project 1 — Hotel Bookings (EDA, Cleaning & Preprocessing)

## What this repo contains
- **Notebook**: `gtc_ml_project1_hotel_bookings.ipynb`
- **README.md** (this file)
- **Dataset**: `hotel_bookings.csv` (original file provided)
- *(Optional)* `hotel_bookings_cleaned.csv` and `X_train/X_test/y_train/y_test` from my pipeline

## Goal
Prepare a clean, ML-ready dataset to **predict booking cancellation** (`is_canceled`) later.  
This repo focuses on **EDA, data quality checks, cleaning, feature engineering, encoding, and a deterministic train/test split**.

## Steps I implemented
### 1) EDA
- `df.info()`, `describe(include=[np.number])`
- Missingness table + bar chart
- Boxplots for `adr` and `lead_time` to spot outliers

### 2) Cleaning
- **Missing values**
  - `company`, `agent` → fill with `0` (kept as Int64)
  - `country` → fill with **mode**
  - `children` → coerced to numeric + fill with **median**
- **Duplicates**: dropped
- **Outliers**
  - `adr > 1000` capped to **1000**
  - negative `adr` capped to **0**
- **Dates**
  - `reservation_status_date` parsed to datetime
  - Built a proper `arrival_date` from year + month (text) + day, then dropped the text month

### 3) Feature Engineering
- `total_guests = adults + children + babies`
- `total_nights = stays_in_weekend_nights + stays_in_week_nights`
- `is_family = 1` if `children>0 or babies>0` else `0`

### 4) Encoding
- **High-cardinality**: `country` → **frequency encoding** to `country_freq`, then dropped text column
- **Low-cardinality**: one-hot for  
  `hotel`, `meal`, `market_segment`, `distribution_channel`,  
  `deposit_type`, `customer_type`, `reserved_room_type`, `assigned_room_type`

### 5) Leakage Removal
- Dropped **`reservation_status`** and **`reservation_status_date`** (contain outcome info)

### 6) Train/Test Split
- Target: **`is_canceled`**
- Deterministic split **80/20** with seed **42** → `X_train`, `X_test`, `y_train`, `y_test`

## How to run (Colab)
1. Open Google Colab → upload this notebook.  
2. Upload `hotel_bookings.csv`:
   ```python
   from google.colab import files
   uploaded = files.upload()
