from datetime import datetime
from pathlib import Path

import polars as pl


CCF_TO_KGAL = 0.7480519
GAL_TO_KGAL = 0.001
MIN_BILL_DAYS = 20.0
MAX_BILL_DAYS = 40.0
MIN_SUMMER_TRIPLE_BILLED_DAILY_AVG_KGAL = 0.3
SUMMER_TO_WINTER_MEAN_RATIO_THRESHOLD = 2.0
MIN_SUMMER_MEAN_NORMALIZED_30DAY_KGAL = 3.0
SUMMER_MONTHS = [4, 5, 6, 7, 8, 9]
SUMMER_START_MONTH = 4
SUMMER_END_MONTH = 10
SUMMER_WEEK_COUNT = 26
SECONDS_PER_DAY = 86400.0

CONSUMPTION_PATH = Path(r"data\consumption_mar2025_mar2026_ccf.csv")
PERSON_PATH = Path(r"data\person_mar2025_mar2026_ccf.csv")
DRC_PATH = Path(r"data\drc_reads_mar2025_mar2026_ccf.csv")
PARCEL_PATH = Path(
    r"\\jeasas2p1\Utility Analytics\Load Research\Projects\jeagis\parcel_enriched_spid_20260306.csv"
)
OUTPUT_PATH = Path(r"data\sp_consumption_summary_20260306_kgal.csv")
SUMMER_WEEKLY_OUTPUT_PATH = Path(r"data\sp_drc_summer_26_weeks_20260306_kgal.csv")

SUPPORTED_SA_TYPES = {"WRES", "WRESSEW", "WRESIRR", "WRECRES"}
DRC_CCF_UOMS = {"CCF"}
DRC_GAL_UOMS = {"GAL"}
DRC_SUPPORTED_UOMS = DRC_CCF_UOMS | DRC_GAL_UOMS

CSV_SCHEMA_OVERRIDES = {
    "BSEG_ID": pl.String,
    "SP_ID": pl.String,
    "SA_ID": pl.String,
    "ACCT_ID": pl.String,
    "PER_ID": pl.String,
    "SA_TYPE_CD": pl.String,
    "CHAR_PREM_ID": pl.String,
    "PREM_ID": pl.String,
    "ADDRESS1": pl.String,
    "FINAL_UOM_CD": pl.String,
    "ENTITY_NAME": pl.String,
    "EMAILID": pl.String,
    "PHONE_TYPE_CD": pl.String,
    "PHONE": pl.String,
    "BEGIN_READ_DTTM": pl.String,
    "CONSUMPTION": pl.Float64,
    "UOM": pl.String,
    "MULTIPLIER": pl.Float64,
}

SUMMARY_COLUMNS = [
    "SP_ID",
    "ACCT_ID",
    "PER_ID",
    "ENTITY_NAME",
    "EMAILID",
    "PHONE_TYPE_CD",
    "PHONE",
    "Type",
    "CHAR_PREM_ID",
    "summer_bill_count",
    "summer_billed_daily_avg_kgal",
    "summer_triple_billed_daily_avg_kgal",
    "summer_mean_normalized_30day_kgal",
    "summer_median_normalized_30day_kgal",
    "summer_drc_coverage_days",
    "summer_drc_above_3x_days",
    "summer_drc_above_3x_intervals",
    "winter_bill_count",
    "winter_mean_normalized_30day_kgal",
    "winter_median_normalized_30day_kgal",
]


def read_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path, null_values="", schema_overrides=CSV_SCHEMA_OVERRIDES, encoding="latin1"
    )
    rename_map = {c: c.upper() for c in df.columns if c != c.upper()}
    return df.rename(rename_map) if rename_map else df


def read_parcel(path: Path) -> pl.DataFrame:
    return (
        pl.read_csv(
            path,
            columns=["CIS_SP_ID", "LND_SQFOOT"],
            null_values="",
            schema_overrides={"CIS_SP_ID": pl.String, "LND_SQFOOT": pl.Float64},
        )
        .select("CIS_SP_ID", "LND_SQFOOT")
        .drop_nulls("CIS_SP_ID")
        .unique(subset=["CIS_SP_ID"], keep="last", maintain_order=True)
    )


def prep_consumption(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select(
            "BSEG_ID",
            "SP_ID",
            "SA_ID",
            "ACCT_ID",
            "SA_TYPE_CD",
            "CHAR_PREM_ID",
            "PREM_ID",
            "ADDRESS1",
            "FINAL_REG_QTY",
            "START_READ_DTTM",
            "END_READ_DTTM",
            "FINAL_UOM_CD",
        )
        .with_columns(
            pl.col("BSEG_ID").cast(pl.String).str.strip_chars(),
            pl.col("SP_ID").cast(pl.String).str.strip_chars(),
            pl.col("SA_ID").cast(pl.String).str.strip_chars(),
            pl.col("ACCT_ID").cast(pl.String).str.strip_chars(),
            pl.col("SA_TYPE_CD").cast(pl.String).str.strip_chars().str.to_uppercase(),
            pl.col("CHAR_PREM_ID").cast(pl.String).str.strip_chars(),
            pl.col("PREM_ID").cast(pl.String).str.strip_chars(),
            pl.col("ADDRESS1").cast(pl.String).str.strip_chars(),
            pl.col("FINAL_REG_QTY").cast(pl.Float64),
            pl.col("FINAL_UOM_CD").cast(pl.String).str.strip_chars().str.to_uppercase(),
            pl.col("START_READ_DTTM").str.strptime(pl.Datetime, strict=True),
            pl.col("END_READ_DTTM").str.strptime(pl.Datetime, strict=True),
        )
        .drop_nulls(
            [
                "BSEG_ID",
                "SP_ID",
                "SA_ID",
                "ACCT_ID",
                "SA_TYPE_CD",
                "FINAL_REG_QTY",
                "START_READ_DTTM",
                "END_READ_DTTM",
            ]
        )
        .unique(subset=["BSEG_ID"], keep="last", maintain_order=True)
        .with_columns(
            (
                (pl.col("END_READ_DTTM") - pl.col("START_READ_DTTM")).dt.total_seconds()
                / 86400.0
            ).alias("bill_days")
        )
        .filter(
            pl.col("bill_days").is_between(MIN_BILL_DAYS, MAX_BILL_DAYS, closed="both")
        )
        .filter(pl.col("SA_TYPE_CD").is_in(SUPPORTED_SA_TYPES))
        .with_columns(
            (pl.col("FINAL_REG_QTY") * CCF_TO_KGAL * 30.0 / pl.col("bill_days")).alias(
                "normalized_30day_kgal"
            ),
            pl.when(pl.col("END_READ_DTTM").dt.month().is_in(SUMMER_MONTHS))
            .then(pl.lit("summer"))
            .otherwise(pl.lit("winter"))
            .alias("season"),
        )
    )


def prep_person(df: pl.DataFrame) -> pl.DataFrame:
    phone_type = (
        pl.col("PHONE_TYPE_CD").cast(pl.String).str.strip_chars().str.to_uppercase()
    )

    return (
        df.select(
            "ACCT_ID", "PER_ID", "ENTITY_NAME", "EMAILID", "PHONE_TYPE_CD", "PHONE"
        )
        .with_columns(
            pl.col("ACCT_ID").cast(pl.String).str.strip_chars(),
            pl.col("PER_ID").cast(pl.String).str.strip_chars(),
            pl.col("ENTITY_NAME").cast(pl.String).str.strip_chars(),
            pl.col("EMAILID").cast(pl.String).str.strip_chars().str.to_lowercase(),
            pl.when(phone_type.is_in(["CELL", "CEL", "MOBILE", "MOB"]))
            .then(pl.lit("CELL"))
            .when(phone_type.is_in(["HOME", "HM"]))
            .then(pl.lit("HOME"))
            .when(phone_type.is_in(["WORK", "WK", "BUS", "BUSINESS"]))
            .then(pl.lit("WORK"))
            .otherwise(phone_type)
            .alias("PHONE_TYPE_CD"),
            pl.col("PHONE").cast(pl.String).str.strip_chars(),
        )
        .drop_nulls(["ACCT_ID", "PER_ID"])
        .unique(subset=["ACCT_ID", "PER_ID"], keep="first", maintain_order=True)
    )


def prep_drc(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select(
            "SP_ID",
            "BEGIN_READ_DTTM",
            "END_READ_DTTM",
            "CONSUMPTION",
            "UOM",
            "MULTIPLIER",
        )
        .with_columns(
            pl.col("SP_ID").cast(pl.String).str.strip_chars(),
            pl.col("BEGIN_READ_DTTM").str.strptime(pl.Datetime, strict=True),
            pl.col("END_READ_DTTM").str.strptime(pl.Datetime, strict=True),
            pl.col("CONSUMPTION").cast(pl.Float64),
            pl.col("UOM").cast(pl.String).str.strip_chars().str.to_uppercase(),
            pl.coalesce([pl.col("MULTIPLIER").cast(pl.Float64), pl.lit(1.0)]).alias(
                "MULTIPLIER"
            ),
        )
        .drop_nulls(["SP_ID", "BEGIN_READ_DTTM", "END_READ_DTTM", "CONSUMPTION"])
        .filter(pl.col("END_READ_DTTM") > pl.col("BEGIN_READ_DTTM"))
        .with_columns(
            (
                (pl.col("END_READ_DTTM") - pl.col("BEGIN_READ_DTTM")).dt.total_seconds()
                / SECONDS_PER_DAY
            ).alias("interval_days"),
            pl.when(pl.col("UOM").is_in(DRC_CCF_UOMS))
            .then(pl.col("CONSUMPTION") * pl.col("MULTIPLIER") * CCF_TO_KGAL)
            .when(pl.col("UOM").is_in(DRC_GAL_UOMS))
            .then(pl.col("CONSUMPTION") * pl.col("MULTIPLIER") * GAL_TO_KGAL)
            .otherwise(None)
            .alias("interval_kgal"),
        )
        .drop_nulls(["interval_days", "interval_kgal"])
        .filter(pl.col("interval_days") > 0)
    )


def resolve_summer_year(usage: pl.DataFrame) -> int | None:
    years = (
        usage.filter(pl.col("season") == "summer")
        .select(pl.col("END_READ_DTTM").dt.year().alias("summer_year"))
        .drop_nulls()
        .unique()
        .sort("summer_year")
        .get_column("summer_year")
        .to_list()
    )
    return years[0] if years else None


def select_candidate_sp_ids(summer: pl.DataFrame, winter: pl.DataFrame) -> pl.DataFrame:
    return (
        summer.select("SP_ID", "summer_mean_normalized_30day_kgal")
        .join(
            winter.select("SP_ID", "winter_mean_normalized_30day_kgal"),
            on="SP_ID",
            how="inner",
        )
        .filter(
            pl.col("summer_mean_normalized_30day_kgal").is_not_null(),
            pl.col("winter_mean_normalized_30day_kgal").is_not_null(),
            pl.col("summer_mean_normalized_30day_kgal")
            > MIN_SUMMER_MEAN_NORMALIZED_30DAY_KGAL,
            pl.col("summer_mean_normalized_30day_kgal")
            >= pl.col("winter_mean_normalized_30day_kgal")
            * SUMMER_TO_WINTER_MEAN_RATIO_THRESHOLD,
        )
        .select("SP_ID")
        .unique()
        .sort("SP_ID")
    )


def build_summer_drc_metrics(
    drc: pl.DataFrame | None, summer_billed: pl.DataFrame, summer_year: int | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    week_columns = [f"week_{week:02d}" for week in range(1, SUMMER_WEEK_COUNT + 1)]
    summer_thresholds = summer_billed.select(
        "SP_ID",
        "summer_billed_daily_avg_kgal",
        "summer_triple_billed_daily_avg_kgal",
    )

    if drc is None or drc.is_empty() or summer_year is None:
        weekly_base = (
            summer_thresholds.with_columns(
                pl.lit(0.0).alias("summer_drc_coverage_days"),
                pl.lit(0.0).alias("summer_drc_above_3x_days"),
                pl.lit(0.0).alias("summer_drc_above_3x_intervals"),
                *[pl.lit(0.0).alias(column) for column in week_columns],
            ).sort("SP_ID")
        )
        empty_metrics = weekly_base.select(
            "SP_ID",
            "summer_drc_coverage_days",
            "summer_drc_above_3x_days",
            "summer_drc_above_3x_intervals",
        )
        return empty_metrics, weekly_base

    summer_start = datetime(summer_year, SUMMER_START_MONTH, 1)
    summer_end = datetime(summer_year, SUMMER_END_MONTH, 1)

    drc_summer = (
        drc.with_columns(
            pl.max_horizontal(
                [pl.col("BEGIN_READ_DTTM"), pl.lit(summer_start, dtype=pl.Datetime)]
            ).alias("summer_begin_dttm"),
            pl.min_horizontal(
                [pl.col("END_READ_DTTM"), pl.lit(summer_end, dtype=pl.Datetime)]
            ).alias("summer_end_dttm"),
        )
        .with_columns(
            (
                (
                    pl.col("summer_end_dttm") - pl.col("summer_begin_dttm")
                ).dt.total_seconds()
                / SECONDS_PER_DAY
            ).alias("summer_overlap_days")
        )
        .filter(pl.col("summer_overlap_days") > 0)
        .join(summer_thresholds, on="SP_ID", how="left")
        .with_columns(
            (
                pl.col("interval_kgal")
                * pl.col("summer_overlap_days")
                / pl.col("interval_days")
            ).alias("summer_interval_kgal"),
            (pl.col("interval_days") < 0.99).alias("is_subdaily_interval"),
            pl.col("summer_begin_dttm").dt.date().alias("summer_begin_date"),
            (
                pl.col("summer_end_dttm") - pl.duration(microseconds=1)
            ).dt.date().alias("summer_last_date"),
        )
        .with_columns(
            (
                pl.col("summer_interval_kgal") / pl.col("summer_overlap_days")
            ).alias("summer_drc_daily_kgal"),
        )
        .with_columns(
            pl.when(pl.col("summer_triple_billed_daily_avg_kgal").is_not_null())
            .then(
                pl.col("summer_drc_daily_kgal")
                > pl.col("summer_triple_billed_daily_avg_kgal")
            )
            .otherwise(pl.lit(False))
            .alias("is_above_3x"),
        )
    )

    drc_daily = (
        drc_summer.with_columns(
            pl.date_ranges(
                "summer_begin_date",
                "summer_last_date",
                interval="1d",
                closed="both",
            ).alias("summer_date")
        )
        .explode("summer_date")
        .with_columns(
            pl.col("summer_date").cast(pl.Datetime).alias("day_start_dttm")
        )
        .with_columns(
            (pl.col("day_start_dttm") + pl.duration(days=1)).alias("day_end_dttm"),
        )
        .with_columns(
            pl.max_horizontal(
                [pl.col("summer_begin_dttm"), pl.col("day_start_dttm")]
            ).alias("day_component_begin_dttm"),
            pl.min_horizontal(
                [pl.col("summer_end_dttm"), pl.col("day_end_dttm")]
            ).alias("day_component_end_dttm"),
        )
        .with_columns(
            (
                (
                    pl.col("day_component_end_dttm")
                    - pl.col("day_component_begin_dttm")
                ).dt.total_seconds()
                / SECONDS_PER_DAY
            ).alias("day_component_days")
        )
        .filter(pl.col("day_component_days") > 0)
        .with_columns(
            (
                pl.col("interval_kgal")
                * pl.col("day_component_days")
                / pl.col("interval_days")
            ).alias("day_component_kgal")
        )
        .group_by(
            "SP_ID",
            "summer_date",
            "summer_billed_daily_avg_kgal",
            "summer_triple_billed_daily_avg_kgal",
        )
        .agg(
            pl.col("is_subdaily_interval").any().alias("has_subdaily_detail"),
            pl.when(pl.col("is_subdaily_interval"))
            .then(pl.col("day_component_kgal"))
            .otherwise(pl.lit(0.0))
            .sum()
            .alias("subdaily_day_kgal"),
            pl.when(~pl.col("is_subdaily_interval"))
            .then(pl.col("day_component_kgal"))
            .otherwise(pl.lit(0.0))
            .sum()
            .alias("daily_day_kgal"),
        )
        .with_columns(
            pl.when(pl.col("has_subdaily_detail"))
            .then(pl.col("subdaily_day_kgal"))
            .otherwise(pl.col("daily_day_kgal"))
            .alias("summer_daily_kgal")
        )
        .with_columns(
            (
                pl.when(pl.col("summer_triple_billed_daily_avg_kgal").is_not_null())
                .then(
                    pl.col("summer_daily_kgal")
                    > pl.col("summer_triple_billed_daily_avg_kgal")
                )
                .otherwise(pl.lit(False))
            )
            .cast(pl.Float64)
            .alias("above_3x_day"),
            (
                (
                    (
                        pl.col("summer_date").cast(pl.Datetime)
                        - pl.lit(summer_start, dtype=pl.Datetime)
                    ).dt.total_seconds()
                    / SECONDS_PER_DAY
                )
                / 7.0
            )
            .floor()
            .cast(pl.Int64)
            .add(pl.lit(1, dtype=pl.Int64))
            .alias("summer_week"),
        )
    )

    drc_metrics = (
        drc_daily.group_by("SP_ID")
        .agg(
            pl.len().cast(pl.Float64).alias("summer_drc_coverage_days"),
            pl.col("above_3x_day").sum().alias("summer_drc_above_3x_days"),
        )
        .join(
            drc_summer.group_by("SP_ID")
            .agg(
                pl.col("is_above_3x")
                .sum()
                .cast(pl.Float64)
                .alias("summer_drc_above_3x_intervals")
            ),
            on="SP_ID",
            how="left",
        )
        .sort("SP_ID")
    )

    weekly_flags = (
        drc_daily.group_by("SP_ID", "summer_week")
        .agg(pl.col("above_3x_day").sum().alias("above_3x_days"))
        .with_columns(
            pl.concat_str(
                [
                    pl.lit("week_"),
                    pl.col("summer_week").cast(pl.String).str.zfill(2),
                ]
            ).alias("summer_week_label")
        )
        .pivot(
            on="summer_week_label",
            index="SP_ID",
            values="above_3x_days",
            aggregate_function="sum",
        )
        .sort("SP_ID")
    )

    for column in week_columns:
        if column not in weekly_flags.columns:
            weekly_flags = weekly_flags.with_columns(pl.lit(0.0).alias(column))

    weekly_flags = weekly_flags.select("SP_ID", *week_columns)

    weekly_base = (
        pl.concat(
            [
                summer_thresholds,
                drc_metrics.join(summer_thresholds, on="SP_ID", how="left").select(
                    "SP_ID",
                    "summer_billed_daily_avg_kgal",
                    "summer_triple_billed_daily_avg_kgal",
                ),
            ],
            how="vertical_relaxed",
        )
        .unique(subset=["SP_ID"], keep="first", maintain_order=True)
        .sort("SP_ID")
    )

    weekly_panel = (
        weekly_base
        .join(drc_metrics, on="SP_ID", how="left")
        .join(weekly_flags, on="SP_ID", how="left")
        .with_columns(
            pl.col("summer_drc_coverage_days").fill_null(0.0),
            pl.col("summer_drc_above_3x_days").fill_null(0.0),
            pl.col("summer_drc_above_3x_intervals").fill_null(0.0),
            *[pl.col(column).fill_null(0.0) for column in week_columns],
        )
        .sort("SP_ID")
    )

    return drc_metrics, weekly_panel


def build_drc_diagnostics(
    raw_drc: pl.DataFrame | None,
    prepared_drc: pl.DataFrame | None,
    summer_billed: pl.DataFrame,
    summer_year: int | None,
    summary: pl.DataFrame,
    candidate_sp_id_count: int,
) -> dict[str, int]:
    diagnostics = {
        "candidate_sp_ids": candidate_sp_id_count,
        "raw_drc_rows": 0,
        "raw_drc_sp_ids": 0,
        "raw_supported_uom_rows": 0,
        "raw_supported_uom_sp_ids": 0,
        "prepared_drc_rows": 0,
        "prepared_drc_sp_ids": 0,
        "summer_overlap_drc_rows": 0,
        "summer_overlap_drc_sp_ids": 0,
        "summer_overlap_with_billed_threshold_sp_ids": 0,
        "summer_overlap_without_billed_threshold_sp_ids": 0,
        "summary_sp_ids": 0,
        "summary_sp_ids_with_any_drc_coverage": 0,
        "summary_sp_ids_with_drc_coverage_gt_170": 0,
    }

    diagnostics["summary_sp_ids"] = summary.select("SP_ID").unique().height
    diagnostics["summary_sp_ids_with_any_drc_coverage"] = (
        summary.filter(pl.col("summer_drc_coverage_days") > 0)
        .select("SP_ID")
        .unique()
        .height
    )
    diagnostics["summary_sp_ids_with_drc_coverage_gt_170"] = (
        summary.filter(pl.col("summer_drc_coverage_days") > 170)
        .select("SP_ID")
        .unique()
        .height
    )

    if raw_drc is None or raw_drc.is_empty():
        return diagnostics

    raw_drc_normalized = raw_drc.select(
        pl.col("SP_ID").cast(pl.String).str.strip_chars().alias("SP_ID"),
        pl.col("UOM").cast(pl.String).str.strip_chars().str.to_uppercase().alias("UOM"),
    )
    diagnostics["raw_drc_rows"] = raw_drc.height
    diagnostics["raw_drc_sp_ids"] = (
        raw_drc_normalized.select("SP_ID").drop_nulls().unique().height
    )

    raw_supported_uom = raw_drc_normalized.filter(pl.col("UOM").is_in(DRC_SUPPORTED_UOMS))
    diagnostics["raw_supported_uom_rows"] = raw_supported_uom.height
    diagnostics["raw_supported_uom_sp_ids"] = (
        raw_supported_uom.select("SP_ID").drop_nulls().unique().height
    )

    if prepared_drc is None or prepared_drc.is_empty():
        return diagnostics

    diagnostics["prepared_drc_rows"] = prepared_drc.height
    diagnostics["prepared_drc_sp_ids"] = prepared_drc.select("SP_ID").unique().height

    if summer_year is None:
        return diagnostics

    summer_start = datetime(summer_year, SUMMER_START_MONTH, 1)
    summer_end = datetime(summer_year, SUMMER_END_MONTH, 1)
    summer_overlap = (
        prepared_drc.with_columns(
            pl.max_horizontal(
                [pl.col("BEGIN_READ_DTTM"), pl.lit(summer_start, dtype=pl.Datetime)]
            ).alias("summer_begin_dttm"),
            pl.min_horizontal(
                [pl.col("END_READ_DTTM"), pl.lit(summer_end, dtype=pl.Datetime)]
            ).alias("summer_end_dttm"),
        )
        .with_columns(
            (
                (
                    pl.col("summer_end_dttm") - pl.col("summer_begin_dttm")
                ).dt.total_seconds()
                / SECONDS_PER_DAY
            ).alias("summer_overlap_days")
        )
        .filter(pl.col("summer_overlap_days") > 0)
        .select("SP_ID")
    )

    diagnostics["summer_overlap_drc_rows"] = summer_overlap.height
    diagnostics["summer_overlap_drc_sp_ids"] = summer_overlap.unique().height

    summer_threshold_sp_ids = summer_billed.select("SP_ID").unique()
    diagnostics["summer_overlap_with_billed_threshold_sp_ids"] = (
        summer_overlap.join(summer_threshold_sp_ids, on="SP_ID", how="inner")
        .unique()
        .height
    )
    diagnostics["summer_overlap_without_billed_threshold_sp_ids"] = (
        summer_overlap.join(summer_threshold_sp_ids, on="SP_ID", how="anti")
        .unique()
        .height
    )

    return diagnostics


def build_summary(
    consumption: pl.DataFrame, person: pl.DataFrame, drc: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    usage = prep_consumption(consumption)
    people = prep_person(person)
    prepared_drc = prep_drc(drc) if drc is not None else None
    summer_year = resolve_summer_year(usage)

    sp_people = (
        usage.select("SP_ID", "ACCT_ID")
        .unique()
        .join(people, on="ACCT_ID", how="inner")
        .select(
            "SP_ID",
            "ACCT_ID",
            "PER_ID",
            "ENTITY_NAME",
            "EMAILID",
            "PHONE_TYPE_CD",
            "PHONE",
        )
        .unique()
    )

    sp_attributes = (
        usage.group_by("SP_ID")
        .agg(
            pl.col("CHAR_PREM_ID")
            .drop_nulls()
            .unique()
            .sort()
            .str.join("|")
            .alias("CHAR_PREM_ID"),
            (pl.col("SA_TYPE_CD") == "WRES").max().alias("has_water"),
            (pl.col("SA_TYPE_CD") == "WRESSEW").max().alias("has_sewer"),
            (pl.col("SA_TYPE_CD") == "WRESIRR").max().alias("has_irrigation"),
            (pl.col("SA_TYPE_CD") == "WRECRES").max().alias("has_reclaimed"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.when(pl.col("has_water"))
                    .then(pl.lit("W"))
                    .otherwise(pl.lit("")),
                    pl.when(pl.col("has_sewer"))
                    .then(pl.lit("S"))
                    .otherwise(pl.lit("")),
                    pl.when(pl.col("has_irrigation"))
                    .then(pl.lit("I"))
                    .otherwise(pl.lit("")),
                    pl.when(pl.col("has_reclaimed"))
                    .then(pl.lit("R"))
                    .otherwise(pl.lit("")),
                ]
            ).alias("Type")
        )
        .select("SP_ID", "Type", "CHAR_PREM_ID")
    )

    bill_type_counts = usage.group_by("SP_ID").agg(
        pl.col("SA_TYPE_CD").n_unique().cast(pl.Float64).alias("bill_type_count")
    )

    summer = (
        usage.filter(pl.col("season") == "summer")
        .group_by("SP_ID")
        .agg(
            pl.len().cast(pl.Float64).alias("summer_bill_rows"),
            (pl.col("FINAL_REG_QTY") * CCF_TO_KGAL).sum().alias("summer_total_kgal"),
            pl.col("bill_days").sum().alias("summer_total_bill_days"),
            pl.col("normalized_30day_kgal")
            .mean()
            .alias("summer_mean_normalized_30day_kgal"),
            pl.col("normalized_30day_kgal")
            .median()
            .alias("summer_median_normalized_30day_kgal"),
        )
        .join(bill_type_counts, on="SP_ID", how="left")
        .with_columns(
            (pl.col("summer_bill_rows") / pl.col("bill_type_count")).alias(
                "summer_bill_count"
            ),
            (pl.col("summer_total_kgal") / pl.col("summer_total_bill_days")).alias(
                "summer_billed_daily_avg_kgal"
            ),
        )
        .with_columns(
            pl.max_horizontal(
                [
                    pl.col("summer_billed_daily_avg_kgal") * 3.0,
                    pl.lit(MIN_SUMMER_TRIPLE_BILLED_DAILY_AVG_KGAL),
                ]
            ).alias("summer_triple_billed_daily_avg_kgal")
        )
        .select(
            "SP_ID",
            "summer_bill_count",
            "summer_billed_daily_avg_kgal",
            "summer_triple_billed_daily_avg_kgal",
            "summer_mean_normalized_30day_kgal",
            "summer_median_normalized_30day_kgal",
        )
    )

    winter = (
        usage.filter(pl.col("season") == "winter")
        .group_by("SP_ID")
        .agg(
            pl.len().cast(pl.Float64).alias("winter_bill_rows"),
            pl.col("normalized_30day_kgal")
            .mean()
            .alias("winter_mean_normalized_30day_kgal"),
            pl.col("normalized_30day_kgal")
            .median()
            .alias("winter_median_normalized_30day_kgal"),
        )
        .join(bill_type_counts, on="SP_ID", how="left")
        .with_columns(
            (pl.col("winter_bill_rows") / pl.col("bill_type_count")).alias(
                "winter_bill_count"
            )
        )
        .select(
            "SP_ID",
            "winter_bill_count",
            "winter_mean_normalized_30day_kgal",
            "winter_median_normalized_30day_kgal",
        )
    )

    candidate_sp_ids = select_candidate_sp_ids(summer, winter)
    sp_people = sp_people.join(candidate_sp_ids, on="SP_ID", how="inner")
    sp_attributes = sp_attributes.join(candidate_sp_ids, on="SP_ID", how="inner")
    summer = summer.join(candidate_sp_ids, on="SP_ID", how="inner")
    winter = winter.join(candidate_sp_ids, on="SP_ID", how="inner")
    if prepared_drc is not None:
        prepared_drc = prepared_drc.join(candidate_sp_ids, on="SP_ID", how="inner")

    drc_metrics, weekly_panel = build_summer_drc_metrics(
        prepared_drc, summer, summer_year
    )

    summary = (
        sp_people.join(sp_attributes, on="SP_ID", how="inner")
        .join(summer, on="SP_ID", how="left")
        .join(drc_metrics, on="SP_ID", how="left")
        .join(winter, on="SP_ID", how="left")
        .with_columns(
            pl.col("summer_drc_coverage_days").fill_null(0.0),
            pl.col("summer_drc_above_3x_days").fill_null(0.0),
            pl.col("summer_drc_above_3x_intervals").fill_null(0.0),
        )
        .with_columns(
            pl.col("summer_billed_daily_avg_kgal").round(4),
            pl.col("summer_triple_billed_daily_avg_kgal").round(4),
            pl.col("summer_mean_normalized_30day_kgal").round(2),
            pl.col("summer_median_normalized_30day_kgal").round(2),
            pl.col("summer_drc_coverage_days").round(2),
            pl.col("summer_drc_above_3x_days").round(2),
            pl.col("summer_drc_above_3x_intervals").round(2),
            pl.col("winter_mean_normalized_30day_kgal").round(2),
            pl.col("winter_median_normalized_30day_kgal").round(2),
        )
        .select(SUMMARY_COLUMNS)
        .sort("SP_ID", "PER_ID", "ACCT_ID")
    )

    weekly_panel = weekly_panel.with_columns(
        pl.col("summer_billed_daily_avg_kgal").round(4),
        pl.col("summer_triple_billed_daily_avg_kgal").round(4),
        pl.col("summer_drc_coverage_days").round(2),
        pl.col("summer_drc_above_3x_days").round(2),
        pl.col("summer_drc_above_3x_intervals").round(2),
        *[pl.col(f"week_{week:02d}").round(2) for week in range(1, SUMMER_WEEK_COUNT + 1)],
    )

    diagnostics = build_drc_diagnostics(
        drc,
        prepared_drc,
        summer,
        summer_year,
        summary,
        candidate_sp_ids.height,
    )

    return summary, weekly_panel, diagnostics


def main() -> None:
    def log(message: str) -> None:
        print(message, flush=True)

    log(f"Reading consumption CSV: {CONSUMPTION_PATH}")
    consumption = read_csv(CONSUMPTION_PATH)
    log(f"Loaded consumption rows={consumption.height}")

    log(f"Reading person CSV: {PERSON_PATH}")
    person = read_csv(PERSON_PATH)
    log(f"Loaded person rows={person.height}")

    log(f"Reading DRC CSV: {DRC_PATH}")
    drc = read_csv(DRC_PATH) if DRC_PATH.exists() else None
    if drc is None:
        log(f"DRC file not found at {DRC_PATH}; DRC coverage metrics defaulted to zero.")
    else:
        log(f"Loaded DRC rows={drc.height}")

    log("Building summary and DRC metrics...")
    summary, weekly_panel, drc_diagnostics = build_summary(consumption, person, drc)
    log("Finished building summary and DRC metrics")

    total_sp_ids = summary.select("SP_ID").unique().height
    drc_covered_sp_ids = (
        summary.filter(pl.col("summer_drc_coverage_days") > 170)
        .select("SP_ID")
        .unique()
        .height
    )

    if PARCEL_PATH.exists():
        log(f"Reading parcel CSV: {PARCEL_PATH}")
        parcel = read_parcel(PARCEL_PATH)
        summary = summary.join(
            parcel, left_on="SP_ID", right_on="CIS_SP_ID", how="left"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing summary CSV: {OUTPUT_PATH}")
    summary.write_csv(OUTPUT_PATH)
    log(f"Writing weekly DRC CSV: {SUMMER_WEEKLY_OUTPUT_PATH}")
    weekly_panel.write_csv(SUMMER_WEEKLY_OUTPUT_PATH)

    log(f"Wrote {OUTPUT_PATH}")
    log(f"Wrote {SUMMER_WEEKLY_OUTPUT_PATH}")
    log(
        "Candidate SP_IDs with summer_mean_normalized_30day_kgal > "
        f"{MIN_SUMMER_MEAN_NORMALIZED_30DAY_KGAL:.1f} and >= "
        f"{SUMMER_TO_WINTER_MEAN_RATIO_THRESHOLD:.1f}x winter mean: "
        f"{drc_diagnostics['candidate_sp_ids']}"
    )
    log(
        "SP_IDs with summer_drc_coverage_days > 170: "
        f"{drc_covered_sp_ids} / {total_sp_ids}"
    )
    log(
        "DRC diagnostics: "
        f"raw_rows={drc_diagnostics['raw_drc_rows']}, "
        f"raw_sp_ids={drc_diagnostics['raw_drc_sp_ids']}, "
        f"raw_supported_uom_rows={drc_diagnostics['raw_supported_uom_rows']}, "
        f"raw_supported_uom_sp_ids={drc_diagnostics['raw_supported_uom_sp_ids']}"
    )
    log(
        "DRC diagnostics: "
        f"prepared_rows={drc_diagnostics['prepared_drc_rows']}, "
        f"prepared_sp_ids={drc_diagnostics['prepared_drc_sp_ids']}, "
        f"summer_overlap_rows={drc_diagnostics['summer_overlap_drc_rows']}, "
        f"summer_overlap_sp_ids={drc_diagnostics['summer_overlap_drc_sp_ids']}"
    )
    log(
        "DRC diagnostics: "
        "summer_overlap_sp_ids_with_billed_threshold="
        f"{drc_diagnostics['summer_overlap_with_billed_threshold_sp_ids']}, "
        "summer_overlap_sp_ids_without_billed_threshold="
        f"{drc_diagnostics['summer_overlap_without_billed_threshold_sp_ids']}, "
        f"summary_sp_ids_with_any_drc_coverage={drc_diagnostics['summary_sp_ids_with_any_drc_coverage']}"
    )


if __name__ == "__main__":
    main()
