# Lab Title: **Advanced BigQuery: Data Transformation and Query Optimization**

This is my modified implementation of Lab 3.
All queries and tables in this lab were implemented within my Google Cloud project gcp-lab3-mlops.
I modified the original Lab 3 by manually uploading the professor-provided CSV into my own BigQuery dataset and building a completely custom processing pipeline around it. I created a cleaned and enriched table with engineered features such as trip duration, start hour, and day of week, and I implemented a brand-new JavaScript UDF to calculate speed in km/h using the haversine formula. I then generated a speed-enhanced table, performed a unique analytics query to identify the fastest routes, and added an advanced SQL window function to compute rolling average speeds—none of which exist in the professor's version. Additionally, I optimized the data warehouse layer using a different partitioning strategy (partition by end date) and clustering column (end station name), and I built a custom materialized view for average duration by hour. These modifications ensure that my implementation is functional, original, and clearly distinct from the provided lab.

The professor's repository included a Bikeshare CSV file.
I manually uploaded this dataset into BigQuery, instead of using the public dataset referenced in the lab instructions.

## Implementation Steps

Below are the steps I completed and the SQL used for each part.

### STEP 1 — Cleaned & Enriched Table

Adds:
duration_min
start_hour
day_of_week
Removes rows with missing timestamps

Query: 
CREATE OR REPLACE TABLE `gcp-lab3-mlops.bikeshare_custom.cleaned_bikeshare` AS
SELECT
  *,
  TIMESTAMP_DIFF(end_time, start_time, MINUTE) AS duration_min,
  EXTRACT(HOUR FROM start_time) AS start_hour,
  EXTRACT(DAYOFWEEK FROM start_time) AS day_of_week
FROM
  `gcp-lab3-mlops.bikeshare_custom.custom_bikeshare`
WHERE
  start_time IS NOT NULL
  AND end_time IS NOT NULL;


### STEP 2 — Create the UDF (JavaScript Function)

Computes speed in km/h from coordinates + duration.

Query: 
CREATE OR REPLACE FUNCTION `gcp-lab3-mlops.bikeshare_custom.calc_speed_kmh`(
  start_lat FLOAT64,
  start_lon FLOAT64,
  end_lat FLOAT64,
  end_lon FLOAT64,
  duration_min FLOAT64
)
RETURNS FLOAT64
LANGUAGE js AS """
  function rad(x){ return x * Math.PI / 180; }
  var R = 6371;  // Earth radius in km

  var dLat = rad(end_lat - start_lat);
  var dLon = rad(end_lon - start_lon);

  var a = Math.sin(dLat/2)**2
        + Math.cos(rad(start_lat)) * Math.cos(rad(end_lat))
        * Math.sin(dLon/2)**2;

  var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  var distance_km = R * c;

  // JS null check
  if (duration_min == null || duration_min <= 0) {
    return null;
  }

  return distance_km / (duration_min / 60);   // km/h
""";


### STEP 3 — Create Table With Speed

Now we apply the UDF to cleaned data.
We also filter out invalid durations (duration_min > 0).

Query: 
CREATE OR REPLACE TABLE `gcp-lab3-mlops.bikeshare_custom.bikeshare_with_speed` AS
SELECT
  *,
  `gcp-lab3-mlops.bikeshare_custom.calc_speed_kmh`(
    start_station_latitude,
    start_station_longitude,
    end_station_latitude,
    end_station_longitude,
    duration_min
  ) AS speed_kmh
FROM
  `gcp-lab3-mlops.bikeshare_custom.cleaned_bikeshare`
WHERE
  duration_min > 0;

### STEP 4 — Custom Analytics Query (Fastest Routes)

Because your dataset is small, we use HAVING trip_count >= 1.

Query: 
SELECT
  start_station_name,
  end_station_name,
  AVG(speed_kmh) AS avg_speed_kmh,
  COUNT(*) AS trip_count
FROM
  `gcp-lab3-mlops.bikeshare_custom.bikeshare_with_speed`
WHERE
  speed_kmh IS NOT NULL
GROUP BY
  start_station_name,
  end_station_name
HAVING trip_count >= 1
ORDER BY avg_speed_kmh DESC
LIMIT 10;

### STEP 5 — Partitioned + Clustered Table

This is one of your modifications:
Partition by end_time (NOT start_time)
Cluster by end_station_name

Query:
CREATE OR REPLACE TABLE `gcp-lab3-mlops.bikeshare_custom.partitioned_clustered_bikeshare`
PARTITION BY DATE(end_time)
CLUSTER BY end_station_name AS
SELECT *
FROM `gcp-lab3-mlops.bikeshare_custom.bikeshare_with_speed`;

### STEP 6 — Materialized View (Custom View)

Average trip duration by hour.

Query:
CREATE MATERIALIZED VIEW `gcp-lab3-mlops.bikeshare_custom.mv_avg_duration_by_hour` AS
SELECT
  start_hour,
  AVG(duration_min) AS avg_duration_min,
  COUNT(*) AS trip_count
FROM
  `gcp-lab3-mlops.bikeshare_custom.bikeshare_with_speed`
GROUP BY
  start_hour;

SELECT *
FROM `gcp-lab3-mlops.bikeshare_custom.mv_avg_duration_by_hour`
ORDER BY start_hour;

### STEP 7 — Validation Queries

Validate speed distribution:
Query:
SELECT
  MIN(speed_kmh) AS min_speed,
  MAX(speed_kmh) AS max_speed,
  AVG(speed_kmh) AS avg_speed
FROM `gcp-lab3-mlops.bikeshare_custom.bikeshare_with_speed`
WHERE speed_kmh IS NOT NULL;

Count valid rows:
Query:
SELECT COUNT(*) 
FROM `gcp-lab3-mlops.bikeshare_custom.bikeshare_with_speed`;

View a sample:
Query:
SELECT *
FROM `gcp-lab3-mlops.bikeshare_custom.bikeshare_with_speed`
LIMIT 10;

### Step 8

WINDOW FUNCTION ANALYSIS
For each row:
It looks at trips from the same start station
Orders them by time
Computes the average speed of the last 4 trips, including the current one

Query:
SELECT
  trip_id,
  start_station_name,
  end_station_name,
  start_time,
  speed_kmh,

  -- Window function: rolling average speed
  AVG(speed_kmh) OVER (
    PARTITION BY start_station_name
    ORDER BY start_time
    ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
  ) AS rolling_avg_speed_last_4_trips

FROM `gcp-lab3-mlops.bikeshare_custom.bikeshare_with_speed`
WHERE speed_kmh IS NOT NULL
ORDER BY start_station_name, start_time;
