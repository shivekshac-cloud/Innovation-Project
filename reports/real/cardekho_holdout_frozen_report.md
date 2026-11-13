# CarDekho hold-out (code frozen at 7a376f4a6226)

Issues found: 12. Issues left: 3.
Cells changed: 72. Rows dropped: 1202.

## Steps

| # | Iter | Step | Status | Changed | Blanked | Why |
|---|---|---|---|---|---|---|
| 1 | 1 | `parse_number(col=year)` | applied | 0 | 0 | 0 values in non-standard formats; column stored as text |
| 2 | 1 | `parse_number(col=selling_price)` | applied | 0 | 0 | 0 values in non-standard formats; column stored as text |
| 3 | 1 | `parse_number(col=km_driven)` | applied | 0 | 0 | 0 values in non-standard formats; column stored as text |
| 4 | 1 | `parse_number(col=seats)` | applied | 0 | 0 | 0 values in non-standard formats; column stored as text |
| 5 | 1 | `normalize_text(col=name, case=None)` | applied | 1 | 0 | 1 values: text format |
| 6 | 1 | `normalize_text(col=torque, case=None)` | applied | 17 | 0 | 17 values: text format |
| 7 | 1 | `map_values(col=max_power, mapping=1 values)` | applied | 1 | 0 | merge 1 variant spellings into canonical values |
| 8 | 1 | `map_values(col=torque, mapping=13 values)` | applied | 53 | 0 | merge 13 variant spellings into canonical values |
| 9 | 1 | `drop_duplicates(subset=['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats'])` | applied | 0 | 0 | 1202 duplicate rows |

## Examples of changes

- **normalize_text(col=name, case=None)**: `Audi A6 2.0 TDI  Design Edition` -> `Audi A6 2.0 TDI Design Edition`
- **normalize_text(col=torque, case=None)**: `135 Nm at 2500  rpm ` -> `135 Nm at 2500 rpm`; `190 Nm at 1750 rpm ` -> `190 Nm at 1750 rpm`; `190 Nm at 1750  rpm ` -> `190 Nm at 1750 rpm`
- **map_values(col=max_power, mapping=1 values)**: ` bhp` -> `bhp`
- **map_values(col=torque, mapping=13 values)**: `400Nm@ 1750-3000rpm` -> `400nm@ 1750-3000rpm`; `113.75Nm@ 4000rpm` -> `113.75nm@ 4000rpm`; `114.73nm@ 4000rpm` -> `114.73Nm@ 4000rpm`
- **drop_duplicates(subset=['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats'])**: `row` -> `dropped`; `row` -> `dropped`; `row` -> `dropped`

## Issues still open

- owner: category_variants (5)
- engine: category_variants (426)
- max_power: category_variants (1298)
