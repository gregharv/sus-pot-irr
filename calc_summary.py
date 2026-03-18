from pathlib import Path

import polars as pl


CCF_TO_KGAL = 0.7480519
MIN_BILL_DAYS = 20.0
MAX_BILL_DAYS = 40.0
SUMMER_MONTHS = [4, 5, 6, 7, 8, 9]

CONSUMPTION_PATH = Path(r"data\consumption_mar2025_mar2026_ccf.csv")
PERSON_PATH = Path(r"data\person_mar2025_mar2026_ccf.csv")
PARCEL_PATH = Path(
    r"\\jeasas2p1\Utility Analytics\Load Research\Projects\jeagis\parcel_enriched_spid_20260306.csv"
)
OUTPUT_PATH = Path(r"data\sp_consumption_summary_20260306_kgal_test.csv")

SUPPORTED_SA_TYPES = {"WRES", "WRESSEW", "WRESIRR", "WRECRES"}

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
    "summer_mean_normalized_30day_kgal",
    "summer_median_normalized_30day_kgal",
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


def build_summary(consumption: pl.DataFrame, person: pl.DataFrame) -> pl.DataFrame:
    usage = prep_consumption(consumption)
    people = prep_person(person)

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
            )
        )
        .select(
            "SP_ID",
            "summer_bill_count",
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

    return (
        sp_people.join(sp_attributes, on="SP_ID", how="inner")
        .join(summer, on="SP_ID", how="left")
        .join(winter, on="SP_ID", how="left")
        .with_columns(
            pl.col("summer_mean_normalized_30day_kgal").round(2),
            pl.col("summer_median_normalized_30day_kgal").round(2),
            pl.col("winter_mean_normalized_30day_kgal").round(2),
            pl.col("winter_median_normalized_30day_kgal").round(2),
        )
        .select(SUMMARY_COLUMNS)
        .sort("SP_ID", "PER_ID", "ACCT_ID")
    )


def main() -> None:
    consumption = read_csv(CONSUMPTION_PATH)
    person = read_csv(PERSON_PATH)

    summary = build_summary(consumption, person)

    if PARCEL_PATH.exists():
        parcel = read_parcel(PARCEL_PATH)
        summary = summary.join(
            parcel, left_on="SP_ID", right_on="CIS_SP_ID", how="left"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(OUTPUT_PATH)

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
