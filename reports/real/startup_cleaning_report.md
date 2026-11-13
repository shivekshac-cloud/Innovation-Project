# Startup funding: cleaning report

Issues found: 20. Issues left: 4.
Cells changed: 7048. Rows dropped: 0.

## Steps

| # | Iter | Step | Status | Changed | Blanked | Why |
|---|---|---|---|---|---|---|
| 1 | 1 | `fix_encoding(col=Date dd/mm/yyyy)` | applied | 1 | 0 | 1 values contain export junk such as \xc2\xa0 |
| 2 | 1 | `fix_encoding(col=Startup Name)` | applied | 27 | 0 | 27 values contain export junk such as \xc2\xa0 |
| 3 | 1 | `fix_encoding(col=Industry Vertical)` | applied | 27 | 0 | 27 values contain export junk such as \xc2\xa0 |
| 4 | 1 | `fix_encoding(col=SubVertical)` | applied | 28 | 0 | 28 values contain export junk such as \xc2\xa0 |
| 5 | 1 | `fix_encoding(col=City  Location)` | applied | 10 | 0 | 10 values contain export junk such as \xc2\xa0 |
| 6 | 1 | `fix_encoding(col=Investors Name)` | applied | 92 | 0 | 92 values contain export junk such as \xc2\xa0 |
| 7 | 1 | `fix_encoding(col=InvestmentnType)` | applied | 31 | 0 | 31 values contain export junk such as \xc2\xa0 |
| 8 | 1 | `fix_encoding(col=Amount in USD)` | applied | 11 | 0 | 11 values contain export junk such as \xc2\xa0 |
| 9 | 1 | `fix_encoding(col=Remarks)` | applied | 5 | 0 | 5 values contain export junk such as \xc2\xa0 |
| 10 | 1 | `standardize_missing(col=Investors Name)` | applied | 12 | 0 | 12 values: disguised missing |
| 11 | 1 | `standardize_missing(col=Amount in USD)` | applied | 11 | 0 | 11 values: disguised missing |
| 12 | 1 | `parse_number(col=Sr No)` | applied | 0 | 0 | 0 values in non-standard formats; column stored as text |
| 13 | 1 | `parse_number(col=Amount in USD)` | applied | 2072 | 0 | 2072 values in non-standard formats; column stored as text |
| 14 | 1 | `parse_date(col=Date dd/mm/yyyy, dayfirst=True)` | applied | 3044 | 1 | 3044 values: date format; day-first: 1747 dates have a day above 12 |
| 15 | 1 | `normalize_text(col=SubVertical, case=None)` | applied | 0 | 0 | 1 values: text format |
| 16 | 1 | `map_values(col=Industry Vertical, mapping=47 values)` | applied | 89 | 0 | merge 47 variant spellings into canonical values |
| 17 | 1 | `map_values(col=SubVertical, mapping=86 values)` | applied | 70 | 0 | merge 86 variant spellings into canonical values |
| 18 | 1 | `map_values(col=City  Location, mapping=13 values)` | applied | 1422 | 0 | merge 13 variant spellings into canonical values |
| 19 | 1 | `map_values(col=InvestmentnType, mapping=13 values)` | applied | 82 | 0 | merge 13 variant spellings into canonical values |
| 20 | 1 | `map_values(col=Remarks, mapping=13 values)` | applied | 14 | 0 | merge 13 variant spellings into canonical values |

## Examples of changes

- **fix_encoding(col=Date dd/mm/yyyy)**: `\\xc2\\xa010/7/2015` -> `10/7/2015`
- **fix_encoding(col=Startup Name)**: `What\\xe2\\x80\\x99s Up Life` -> `What’s Up Life`; `Byju\\xe2\\x80\\x99s` -> `Byju’s`; `Creator\\xe2\\x80\\x99s Gurukul` -> `Creator’s Gurukul`
- **fix_encoding(col=Industry Vertical)**: `Hyperlocal\\xc2\\xa0 Grocery Delivery Service` -> `Hyperlocal Grocery Delivery Service`; `WiFi \xe2\x80\x93 first Cloud communication platform` -> `WiFi – first Cloud communication platform`; `Mobile Growth Hacking\\nPlatform` -> `Mobile Growth Hacking Platform`
- **fix_encoding(col=SubVertical)**: `News and ebooks Mobile App\\xc2\\xa0` -> `News and ebooks Mobile App`; `Applied Artificial Intelligence and IoT focused\\xc2\\xa0Platform` -> `Applied Artificial Intelligence and IoT focused Platform`; `\\xc2\\xa0Customer Sourcing Platform\\xc2\\xa0` -> `Customer Sourcing Platform`
- **fix_encoding(col=City  Location)**: `\\xc2\\xa0Noida` -> `Noida`; `\\xc2\\xa0Bangalore` -> `Bangalore`; `\\xc2\\xa0Gurgaon` -> `Gurgaon`
- **fix_encoding(col=Investors Name)**: `India Quotient and LetsVenture\\xe2\\x80\\x99s Angel Fund` -> `India Quotient and LetsVenture’s Angel Fund`; `Sequoia India, Hillhouse Capital, Alphabet\\xe2\\x80\\x99s growth investment arm Capital G and Axis Bank` -> `Sequoia India, Hillhouse Capital, Alphabet’s growth investment arm Capital G and Axis Bank`; `Norwest Venture Partners, CDC Group, the UK\\xe2\\x80\\x99s Development Finance Institution and P Surendra Pai` -> `Norwest Venture Partners, CDC Group, the UK’s Development Finance Institution and P Surendra Pai`
- **fix_encoding(col=InvestmentnType)**: `Seed\\nFunding` -> `Seed Funding`; `Seed\\nFunding` -> `Seed Funding`; `Seed\\nFunding` -> `Seed Funding`
- **fix_encoding(col=Amount in USD)**: `\\xc2\\xa020,000,000` -> `20,000,000`; `\\xc2\\xa016,200,000` -> `16,200,000`; `\\xc2\\xa0N/A` -> `N/A`
- **fix_encoding(col=Remarks)**: `Series C\\n(More here)` -> `Series C (More here)`; `\\xc2\\xa0Series B` -> `Series B`; `\\xc2\\xa0Late Stage` -> `Late Stage`
- **standardize_missing(col=Investors Name)**: `Undisclosed` -> `nan`; `Undisclosed` -> `nan`; `Undisclosed` -> `nan`
- **standardize_missing(col=Amount in USD)**: `undisclosed` -> `nan`; `unknown` -> `nan`; `Undisclosed` -> `nan`
- **parse_number(col=Amount in USD)**: `20,00,00,000` -> `200000000.0`; `80,48,394` -> `8048394.0`; `1,83,58,860` -> `18358860.0`
- **parse_date(col=Date dd/mm/yyyy, dayfirst=True)**: `09/01/2020` -> `2020-01-09`; `13/01/2020` -> `2020-01-13`; `09/01/2020` -> `2020-01-09`
- **map_values(col=Industry Vertical, mapping=47 values)**: `Health and wellness` -> `Health and Wellness`; `Saas` -> `SaaS`; `Fintech` -> `FinTech`
- **map_values(col=SubVertical, mapping=86 values)**: `Non-banking financial company` -> `Non-Banking Financial Company`; `Online lending Platform` -> `Online Lending Platform`; `Online Lending Platform For SME` -> `Online Lending Platform for SME`
- **map_values(col=City  Location, mapping=13 values)**: `Gurgaon` -> `Gurugram`; `New Delhi` -> `Delhi`; `Gurgaon` -> `Gurugram`
- **map_values(col=InvestmentnType, mapping=13 values)**: `Pre-series A` -> `Pre-Series A`; `pre-series A` -> `Pre-Series A`; `pre-Series A` -> `Pre-Series A`
- **map_values(col=Remarks, mapping=13 values)**: `pre-Series A` -> `Pre-Series A`; `Bridge funding` -> `Bridge Funding`; `Pre-Series A Bridge round` -> `Pre-Series A bridge round`

## Issues still open

- SubVertical: category_variants (1)
- City  Location: category_variants (115)
- InvestmentnType: category_variants (58)
- Remarks: category_variants (51)
