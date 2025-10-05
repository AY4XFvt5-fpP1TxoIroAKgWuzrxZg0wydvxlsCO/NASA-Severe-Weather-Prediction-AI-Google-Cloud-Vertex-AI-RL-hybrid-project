-- bq/build_sequence_features.sql
-- Builds storm sequence windows for ML model input.

CREATE OR REPLACE TABLE `PROJECT.sar_analysis.sequence_features` AS
WITH storms AS (
  SELECT
    storm_id,
    TIMESTAMP_TRUNC(iso_time, HOUR) AS ts,
    lat,
    lon,
    wind_1min_max,
    pressure_msl
  FROM `PROJECT.DATASET.RAW_STORM_TABLE`
),

sar_feats AS (
  SELECT
    s.storm_id, s.ts,
    AVG(sr.backscatter) AS mean_backscatter
  FROM storms s
  LEFT JOIN `PROJECT.DATASET.SAR_TABLE` sr
    ON ABS(TIMESTAMP_DIFF(s.ts, sr.acq_time, HOUR)) < 72
    AND ST_DWithin(ST_GEOGPOINT(s.lon, s.lat), sr.geometry, 10000)
  GROUP BY s.storm_id, s.ts
),

rean AS (
  SELECT
    ts, lat, lon, sst, precip_24h, u700 AS steering_u, v700 AS steering_v
  FROM `PROJECT.DATASET.REANALYSIS_TABLE`
),

joined AS (
  SELECT
    s.storm_id,
    s.ts,
    s.lat,
    s.lon,
    s.wind_1min_max,
    s.pressure_msl,
    COALESCE(sf.mean_backscatter, 0.0) AS mean_backscatter,
    COALESCE(r.sst, 0.0) AS sst,
    COALESCE(r.precip_24h, 0.0) AS precip_24h,
    COALESCE(r.steering_u, 0.0) AS steering_u,
    COALESCE(r.steering_v, 0.0) AS steering_v
  FROM storms s
  LEFT JOIN sar_feats sf USING (storm_id, ts)
  LEFT JOIN rean r
    ON r.ts = s.ts
    AND ST_DWithin(ST_GEOGPOINT(s.lon, s.lat), ST_GEOGPOINT(r.lon, r.lat), 25000)
),

seq_prep AS (
  SELECT
    storm_id,
    ts,
    ARRAY_AGG(STRUCT(lat, lon, wind_1min_max, pressure_msl, mean_backscatter,
                      sst, precip_24h, steering_u, steering_v)
              ORDER BY ts DESC LIMIT 6) AS seq_window
  FROM joined
  GROUP BY storm_id, ts
  HAVING ARRAY_LENGTH(seq_window) = 6
)

SELECT
  GENERATE_UUID() AS sequence_id,
  storm_id,
  ts AS target_time,
  seq_window
FROM seq_prep;
