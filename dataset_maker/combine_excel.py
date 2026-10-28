
from pathlib import Path
from collections import Counter
import pandas as pd
import re

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "All_India_Old_Age_Homes.xlsx"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_HEADERS = [
    "S.No",
    "State",
    "District",
    "City / Taluka",
    "P.O./ Locality / Village",
    "Home Name",
    "Address",
    "Pin Code",
    "Bed Capacity",
    "Facility Type",
    "Telephone",
    "Mgmt",
    "Gender",
    "Care Type",
    "Source",
]

AUDIT_COLUMNS = ["Source_File", "Source_Sheet", "Source_Row"]

ALIASES = {
    "s.no": "S.No",
    "s.no.": "S.No",
    "s no": "S.No",
    "serial no": "S.No",
    "serial number": "S.No",
    "state": "State",
    "district": "District",
    "city": "City / Taluka",
    "taluka": "City / Taluka",
    "city/taluka": "City / Taluka",
    "city / taluka": "City / Taluka",
    "p.o./ locality / village": "P.O./ Locality / Village",
    "p.o. / locality / village": "P.O./ Locality / Village",
    "po/locality/village": "P.O./ Locality / Village",
    "locality": "P.O./ Locality / Village",
    "village": "P.O./ Locality / Village",
    "home name": "Home Name",
    "name of home": "Home Name",
    "old age home name": "Home Name",
    "address": "Address",
    "pin code": "Pin Code",
    "pincode": "Pin Code",
    "pin": "Pin Code",
    "bed capacity": "Bed Capacity",
    "facility type": "Facility Type",
    "telephone": "Telephone",
    "phone": "Telephone",
    "contact number": "Telephone",
    "mgmt": "Mgmt",
    "management": "Mgmt",
    "gender": "Gender",
    "care type": "Care Type",
    "source": "Source",
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_header(value):
    cleaned = clean_text(value).lower()
    return ALIASES.get(cleaned, cleaned)


def find_header_row(raw, search_limit=50):
    """Find the row containing the actual table headings."""
    for row_idx in range(min(search_limit, len(raw))):
        values = [
            normalize_header(value)
            for value in raw.iloc[row_idx].tolist()
        ]

        matches = sum(
            value in EXPECTED_HEADERS
            for value in values
        )

        if matches >= 5:
            return row_idx

    return None


def make_unique_headers(values):
    """Avoid overwriting columns with duplicate headings."""
    result = []
    counts = Counter()

    for index, value in enumerate(values, start=1):
        original = clean_text(value)
        normalized = normalize_header(value)

        if normalized in EXPECTED_HEADERS:
            name = normalized
        else:
            name = original or f"Unnamed_Column_{index}"

        counts[name] += 1

        if counts[name] > 1:
            name = f"{name}__duplicate_{counts[name]}"

        result.append(name)

    return result


# ============================================================
# 3. DISCOVER EXCEL FILES IN THE PROJECT FOLDER
# ============================================================

excel_files = sorted(
    file
    for file in INPUT_DIR.iterdir()
    if file.is_file()
    and file.suffix.lower() in {".xlsx", ".xlsm", ".xls"}
    and not file.name.startswith("~$")
    and file.resolve() != OUTPUT_FILE.resolve()
)

if not excel_files:
    raise FileNotFoundError(
        f"No Excel files found in: {INPUT_DIR}"
    )

print(f"Project folder: {INPUT_DIR}")
print(f"Excel files found: {len(excel_files)}")


# ============================================================
# 4. READ ALL WORKBOOKS AND WORKSHEETS
# ============================================================

combined_sheets = []
file_summary = []
unparsed_cells = []
preamble_rows = []
unexpected_values = []

for file_path in excel_files:
    print(f"\nProcessing: {file_path.name}")

    try:
        workbook = pd.read_excel(
            file_path,
            sheet_name=None,
            header=None,
            dtype=object,
        )
    except Exception as error:
        file_summary.append({
            "Source_File": file_path.name,
            "Source_Sheet": "",
            "Status": "READ_ERROR",
            "Rows_Combined": 0,
            "Details": str(error),
        })
        print(f"  ERROR: {error}")
        continue

    for sheet_name, raw in workbook.items():

        # Remove fully empty columns only.
        raw = raw.dropna(axis=1, how="all")

        if raw.empty:
            file_summary.append({
                "Source_File": file_path.name,
                "Source_Sheet": sheet_name,
                "Status": "EMPTY_SHEET",
                "Rows_Combined": 0,
                "Details": "No non-empty cells",
            })
            continue

        header_idx = find_header_row(raw)

        if header_idx is None:
            # Preserve all non-empty cells from unrecognized sheets.
            for row_idx, row in raw.iterrows():
                for col_idx, value in row.items():
                    if pd.notna(value):
                        unparsed_cells.append({
                            "Source_File": file_path.name,
                            "Source_Sheet": sheet_name,
                            "Source_Row": int(row_idx) + 1,
                            "Source_Column": int(col_idx) + 1,
                            "Value": value,
                        })

            file_summary.append({
                "Source_File": file_path.name,
                "Source_Sheet": sheet_name,
                "Status": "HEADER_NOT_FOUND",
                "Rows_Combined": 0,
                "Details": "See Unparsed_Sheets",
            })

            print(f"  Header not found: {sheet_name}")
            continue

        # Keep non-empty introductory rows above the table header.
        for row_idx in range(header_idx):
            values = raw.iloc[row_idx].tolist()

            if any(pd.notna(value) for value in values):
                preamble_rows.append({
                    "Source_File": file_path.name,
                    "Source_Sheet": sheet_name,
                    "Source_Row": row_idx + 1,
                    "Content": " | ".join(
                        str(value)
                        for value in values
                        if pd.notna(value)
                    ),
                })

        headers = make_unique_headers(
            raw.iloc[header_idx].tolist()
        )

        # Keep the original Excel row indices for source tracking.
        data = raw.iloc[header_idx + 1:].copy()
        data.columns = headers

        # Remove only fully empty rows; preserve partial records.
        data = data.loc[data.notna().any(axis=1)].copy()

        # FIX: use the retained DataFrame's original index.
        # Excel rows are one-based, hence index + 1.
        source_row_numbers = data.index + 1

        # Record values from any non-standard columns before
        # creating missing standard columns.
        for column in data.columns:
            if column not in EXPECTED_HEADERS:
                for row_idx, value in data[column].items():
                    if pd.notna(value):
                        unexpected_values.append({
                            "Source_File": file_path.name,
                            "Source_Sheet": sheet_name,
                            "Source_Row": int(row_idx) + 1,
                            "Column": column,
                            "Value": value,
                        })

        # Add any missing standard columns as blanks.
        # Do not replace or overwrite existing source data.
        for column in EXPECTED_HEADERS:
            if column not in data.columns:
                data[column] = pd.NA

        # Add traceability metadata.
        data["Source_File"] = file_path.name
        data["Source_Sheet"] = sheet_name
        data["Source_Row"] = source_row_numbers

        extra_columns = [
            column
            for column in data.columns
            if column not in EXPECTED_HEADERS + AUDIT_COLUMNS
        ]

        data = data[
            EXPECTED_HEADERS + extra_columns + AUDIT_COLUMNS
        ]

        combined_sheets.append(data)

        file_summary.append({
            "Source_File": file_path.name,
            "Source_Sheet": sheet_name,
            "Status": "COMBINED",
            "Rows_Combined": len(data),
            "Details": f"Header found on Excel row {header_idx + 1}",
        })

        print(f"  Sheet: {sheet_name} | Rows: {len(data)}")


# ============================================================
# 5. COMBINE ALL RECOGNIZED DATA
# ============================================================

if not combined_sheets:
    raise RuntimeError(
        "No worksheets were combined. Check File_Summary "
        "and Unparsed_Sheets."
    )

master = pd.concat(
    combined_sheets,
    ignore_index=True,
    sort=False,
)

extra_columns = [
    column
    for column in master.columns
    if column not in EXPECTED_HEADERS + AUDIT_COLUMNS
]

master = master[
    EXPECTED_HEADERS + extra_columns + AUDIT_COLUMNS
]


# ============================================================
# 6. WRITE THE OUTPUT EXCEL WORKBOOK
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl",
) as writer:

    master.to_excel(
        writer,
        sheet_name="Master_Data",
        index=False,
    )

    pd.DataFrame(file_summary).to_excel(
        writer,
        sheet_name="File_Summary",
        index=False,
    )

    pd.DataFrame(unparsed_cells).to_excel(
        writer,
        sheet_name="Unparsed_Sheets",
        index=False,
    )

    pd.DataFrame(preamble_rows).to_excel(
        writer,
        sheet_name="Notes_and_Preambles",
        index=False,
    )

    pd.DataFrame(unexpected_values).to_excel(
        writer,
        sheet_name="Unexpected_Columns",
        index=False,
    )

    # Readability settings only; source cell values are unchanged.
    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for cells in worksheet.iter_cols():
            letter = cells[0].column_letter

            max_length = max(
                (
                    len(str(cell.value))
                    for cell in cells
                    if cell.value is not None
                ),
                default=10,
            )

            worksheet.column_dimensions[letter].width = min(
                max(max_length + 2, 12), 40
            )


# ============================================================
# 7. FINAL REPORT
# ============================================================

combined_count = sum(
    item["Rows_Combined"]
    for item in file_summary
    if item["Status"] == "COMBINED"
)

problem_count = sum(
    item["Status"] in {"READ_ERROR", "HEADER_NOT_FOUND"}
    for item in file_summary
)

print("\n" + "=" * 60)
print("ALL-INDIA OLD-AGE-HOME DATASET")
print("=" * 60)
print(f"Excel files discovered : {len(excel_files)}")
print(f"Worksheets combined    : {len(combined_sheets)}")
print(f"Total records combined : {len(master)}")
print(f"Rows reported combined : {combined_count}")
print(f"Standard columns       : {len(EXPECTED_HEADERS)}")
print(f"Extra columns retained : {len(extra_columns)}")
print(f"Sheets needing review  : {problem_count}")
print(f"\nOutput file:\n{OUTPUT_FILE}")
print("\nReview File_Summary and Unparsed_Sheets.")
