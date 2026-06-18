# Fundamentals data dictionary

| column | dtype | unit_category | description | status | non_null | total | pct_populated | files_with_data | pct_files_with_data | example_value | documented |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symbol | string | id | Ticker symbol the row belongs to. | kept | 47086 | 47086 | 100.0 | 493 | 100.0 | AA | True |
| frequency | string | category | Reporting cadence of the statement (quarterly/annual). | constant | 47086 | 47086 | 100.0 | 493 | 100.0 | quarterly | True |
| fiscaldateending | date | date | Last day of the fiscal period the figures cover. | kept | 47086 | 47086 | 100.0 | 493 | 100.0 | 1998-03-31 | True |
| report_date | date | date | Date the earnings/results were publicly reported. | kept | 47086 | 47086 | 100.0 | 493 | 100.0 | 1998-03-31 | True |
| reporttime | string | category | Timing of the earnings release (pre-market/post-market). | kept | 45217 | 47086 | 96.03 | 493 | 100.0 | pre-market | True |
| reportedcurrency | string | category | Currency the financial statement is reported in. | kept | 35644 | 47086 | 75.7 | 493 | 100.0 | USD | True |
| reported_eps | numeric | currency/share | Actual reported earnings per share for the quarter. | kept | 45213 | 47086 | 96.02 | 493 | 100.0 | 0.9298 | True |
| estimated_eps | numeric | currency/share | Consensus analyst estimated EPS for the quarter. | kept | 42982 | 47086 | 91.28 | 493 | 100.0 | 0.04 | True |
| earnings_surprise | numeric | currency/share | Reported EPS minus estimated EPS (absolute surprise). | kept | 45217 | 47086 | 96.03 | 493 | 100.0 | 0 | True |
| surprise_pct | numeric | percent | Earnings surprise as a percentage of the estimate. | kept | 42550 | 47086 | 90.37 | 491 | 99.59 | -50 | True |
| accumulateddepreciationamortizationppe | numeric | currency | Accumulated depreciation/amortization on PP&E. | kept | 3050 | 47086 | 6.48 | 143 | 29.01 | -13536000000 | True |
| capitalexpenditures | numeric | currency | Capital expenditures (capex). | kept | 35625 | 47086 | 75.66 | 493 | 100.0 | 444000000 | True |
| capitalleaseobligations | numeric | currency | Capital/finance lease obligations. | kept | 9829 | 47086 | 20.87 | 404 | 81.95 | 100000000 | True |
| cashandcashequivalentsatcarryingvalue | numeric | currency | Cash and cash equivalents at carrying value. | kept | 35207 | 47086 | 74.77 | 493 | 100.0 | 266000000 | True |
| cashandshortterminvestments | numeric | currency | Cash plus short-term investments. | kept | 35207 | 47086 | 74.77 | 493 | 100.0 | 266000000 | True |
| cashflowfromfinancing | numeric | currency | Net cash from financing activities. | kept | 34974 | 47086 | 74.28 | 493 | 100.0 | -444000000 | True |
| cashflowfrominvestment | numeric | currency | Net cash from investing activities. | kept | 34579 | 47086 | 73.44 | 493 | 100.0 | -338000000 | True |
| changeincashandcashequivalents | numeric | currency | Net change in cash and equivalents. | kept | 17247 | 47086 | 36.63 | 454 | 92.09 | -49000000 | True |
| changeinexchangerate | numeric | currency | FX effect on cash balances. | kept | 6462 | 47086 | 13.72 | 255 | 51.72 | 6000000 | True |
| changeininventory | numeric | currency | Change in inventory (cash flow adjustment). | kept | 28425 | 47086 | 60.37 | 488 | 98.99 | -126000000 | True |
| changeinoperatingassets | numeric | currency | Change in operating assets. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| changeinoperatingliabilities | numeric | currency | Change in operating liabilities. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| changeinreceivables | numeric | currency | Change in receivables. | kept | 13541 | 47086 | 28.76 | 408 | 82.76 | 7000000 | True |
| commonstock | numeric | currency | Common stock at par/stated value. | kept | 34356 | 47086 | 72.96 | 493 | 100.0 | 11915000000 | True |
| commonstocksharesoutstanding | numeric | shares | Number of common shares outstanding. | kept | 35380 | 47086 | 75.14 | 493 | 100.0 | 247343000 | True |
| comprehensiveincomenetoftax | numeric | currency | Comprehensive income net of tax. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| costofgoodsandservicessold | numeric | currency | Direct cost of goods and services sold (COGS). | kept | 35379 | 47086 | 75.14 | 493 | 100.0 | 10548000000 | True |
| costofrevenue | numeric | currency | Total cost of generating revenue. | kept | 35379 | 47086 | 75.14 | 493 | 100.0 | 10548000000 | True |
| currentaccountspayable | numeric | currency | Accounts payable due within a year. | kept | 34504 | 47086 | 73.28 | 493 | 100.0 | 1740000000 | True |
| currentdebt | numeric | currency | Debt due within a year. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| currentlongtermdebt | numeric | currency | Current portion of long-term debt. | kept | 14095 | 47086 | 29.93 | 482 | 97.77 | 29000000 | True |
| currentnetreceivables | numeric | currency | Net receivables due within a year. | kept | 33972 | 47086 | 72.15 | 490 | 99.39 | 712000000 | True |
| deferredrevenue | numeric | currency | Revenue received but not yet earned. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| depreciation | numeric | currency | Depreciation expense for the period. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| depreciationandamortization | numeric | currency | Combined depreciation and amortization expense. | kept | 34633 | 47086 | 73.55 | 493 | 100.0 | 204000000 | True |
| depreciationdepletionandamortization | numeric | currency | Depreciation, depletion & amortization (cash flow). | kept | 34399 | 47086 | 73.06 | 493 | 100.0 | 954000000 | True |
| dividendpayout | numeric | currency | Total dividends paid. | kept | 27240 | 47086 | 57.85 | 482 | 97.77 | 0 | True |
| dividendpayoutcommonstock | numeric | currency | Dividends paid on common stock. | kept | 27240 | 47086 | 57.85 | 482 | 97.77 | 0 | True |
| dividendpayoutpreferredstock | numeric | currency | Dividends paid on preferred stock. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| ebit | numeric | currency | Earnings before interest and taxes. | kept | 35476 | 47086 | 75.34 | 493 | 100.0 | 1216000000 | True |
| ebitda | numeric | currency | Earnings before interest, taxes, depreciation & amortization. | kept | 35067 | 47086 | 74.47 | 493 | 100.0 | 616000000 | True |
| goodwill | numeric | currency | Goodwill from acquisitions. | kept | 29435 | 47086 | 62.51 | 482 | 97.77 | 160000000 | True |
| grossprofit | numeric | currency | Revenue less cost of goods/services sold. | kept | 35554 | 47086 | 75.51 | 493 | 100.0 | 2599000000 | True |
| incomebeforetax | numeric | currency | Pre-tax income (EBT). | kept | 35664 | 47086 | 75.74 | 493 | 100.0 | -63000000 | True |
| incometaxexpense | numeric | currency | Income tax expense/provision. | kept | 35521 | 47086 | 75.44 | 493 | 100.0 | 284000000 | True |
| intangibleassets | numeric | currency | Total intangible assets (incl. goodwill). | kept | 30241 | 47086 | 64.23 | 485 | 98.38 | 230000000 | True |
| intangibleassetsexcludinggoodwill | numeric | currency | Intangible assets excluding goodwill. | kept | 30241 | 47086 | 64.23 | 485 | 98.38 | 230000000 | True |
| interestanddebtexpense | numeric | currency | Combined interest and debt-related expense. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| interestexpense | numeric | currency | Expense incurred on debt/interest obligations. | kept | 34609 | 47086 | 73.5 | 490 | 99.39 | 343000000 | True |
| interestincome | numeric | currency | Income earned from interest-bearing assets. | kept | 10886 | 47086 | 23.12 | 479 | 97.16 | 50000000 | True |
| inventory | numeric | currency | Inventory carried on the balance sheet. | kept | 32067 | 47086 | 68.1 | 489 | 99.19 | 1501000000 | True |
| investmentincomenet | numeric | currency | Net income earned from investments. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| investments | numeric | currency | Total investments held. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| longtermdebt | numeric | currency | Long-term debt. | kept | 30123 | 47086 | 63.97 | 483 | 97.97 | 313000000 | True |
| longtermdebtnoncurrent | numeric | currency | Non-current portion of long-term debt. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| longterminvestments | numeric | currency | Long-term investments. | kept | 14449 | 47086 | 30.69 | 404 | 81.95 | 1777000000 | True |
| netincome | numeric | currency | Bottom-line net income (profit) for the period. | kept | 35838 | 47086 | 76.11 | 493 | 100.0 | -256000000 | True |
| netincomefromcontinuingoperations | numeric | currency | Net income from continuing operations. | kept | 33697 | 47086 | 71.56 | 492 | 99.8 | -347000000 | True |
| netinterestincome | numeric | currency | Interest income net of interest expense. | kept | 13236 | 47086 | 28.11 | 480 | 97.36 | -30000000 | True |
| noninterestincome | numeric | currency | Income from sources other than interest (fees, trading). | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| operatingcashflow | numeric | currency | Net cash generated by operating activities. | kept | 35238 | 47086 | 74.84 | 493 | 100.0 | 842000000 | True |
| operatingexpenses | numeric | currency | Total operating expenses. | kept | 35671 | 47086 | 75.76 | 493 | 100.0 | 11931000000 | True |
| operatingincome | numeric | currency | Profit from core operations (revenue less operating costs). | kept | 35701 | 47086 | 75.82 | 493 | 100.0 | 1167000000 | True |
| othercurrentassets | numeric | currency | Other current assets. | kept | 34514 | 47086 | 73.3 | 493 | 100.0 | 438000000 | True |
| othercurrentliabilities | numeric | currency | Other current liabilities. | kept | 34860 | 47086 | 74.03 | 493 | 100.0 | 825000000 | True |
| othernoncurrentassets | numeric | currency | Other non-current assets. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| othernoncurrentliabilities | numeric | currency | Other non-current liabilities. | kept | 22183 | 47086 | 47.11 | 480 | 97.36 | 3902000000 | True |
| othernonoperatingincome | numeric | currency | Income/expense outside normal operations. | kept | 13553 | 47086 | 28.78 | 407 | 82.56 | 106000000 | True |
| paymentsforoperatingactivities | numeric | currency | Cash payments for operating activities. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| paymentsforrepurchaseofcommonstock | numeric | currency | Cash paid to repurchase common stock. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| paymentsforrepurchaseofequity | numeric | currency | Cash paid to repurchase equity. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| paymentsforrepurchaseofpreferredstock | numeric | currency | Cash paid to repurchase preferred stock. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| proceedsfromissuanceofcommonstock | numeric | currency | Proceeds from issuing common stock. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| proceedsfromissuanceoflongtermdebtandcapitalsecuritiesnet | numeric | currency | Net proceeds from issuing long-term debt/capital securities. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| proceedsfromissuanceofpreferredstock | numeric | currency | Proceeds from issuing preferred stock. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| proceedsfromoperatingactivities | numeric | currency | Cash proceeds from operating activities. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| proceedsfromrepaymentsofshorttermdebt | numeric | currency | Net proceeds/repayments of short-term debt. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| proceedsfromrepurchaseofequity | numeric | currency | Net proceeds/payments from equity repurchase. | kept | 28953 | 47086 | 61.49 | 490 | 99.39 | 10000000 | True |
| proceedsfromsaleoftreasurystock | numeric | currency | Proceeds from sale of treasury stock. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| profitloss | numeric | currency | Profit/loss figure used in the cash flow statement. | dropped_all_null | 0 | 47086 | 0.0 | 0 | 0.0 |  | True |
| propertyplantequipment | numeric | currency | Property, plant & equipment (net PP&E). | kept | 29531 | 47086 | 62.72 | 487 | 98.78 | 11326000000 | True |
| researchanddevelopment | numeric | currency | Research & development expenditure. | kept | 26397 | 47086 | 56.06 | 474 | 96.15 | 95000000 | True |
| retainedearnings | numeric | currency | Cumulative retained earnings. | kept | 34516 | 47086 | 73.3 | 493 | 100.0 | -6000000 | True |
| sellinggeneralandadministrative | numeric | currency | Selling, general & administrative expenses (SG&A). | kept | 34640 | 47086 | 73.57 | 493 | 100.0 | 478000000 | True |
| shortlongtermdebttotal | numeric | currency | Total short- plus long-term debt. | kept | 32753 | 47086 | 69.56 | 493 | 100.0 | 225000000 | True |
| shorttermdebt | numeric | currency | Short-term debt. | kept | 31849 | 47086 | 67.64 | 493 | 100.0 | 29000000 | True |
| shortterminvestments | numeric | currency | Short-term / marketable investments. | kept | 26044 | 47086 | 55.31 | 491 | 99.59 | 1228000000 | True |
| stockbasedcompensation | numeric | currency | Stock-based compensation expense. | kept | 30398 | 47086 | 64.56 | 491 | 99.59 | 11000000 | True |
| totalassets | numeric | currency | Total assets. | kept | 35234 | 47086 | 74.83 | 493 | 100.0 | 18680000000 | True |
| totalcurrentassets | numeric | currency | Assets expected to convert to cash within a year. | kept | 35016 | 47086 | 74.37 | 493 | 100.0 | 2917000000 | True |
| totalcurrentliabilities | numeric | currency | Obligations due within a year. | kept | 35118 | 47086 | 74.58 | 493 | 100.0 | 2735000000 | True |
| totalliabilities | numeric | currency | Total liabilities. | kept | 35189 | 47086 | 74.73 | 493 | 100.0 | 5607000000 | True |
| totalnoncurrentassets | numeric | currency | Long-term (non-current) assets. | kept | 34813 | 47086 | 73.93 | 493 | 100.0 | 13847000000 | True |
| totalnoncurrentliabilities | numeric | currency | Long-term (non-current) liabilities. | kept | 34572 | 47086 | 73.42 | 493 | 100.0 | 2496000000 | True |
| totalrevenue | numeric | currency | Total revenue / net sales for the period. | kept | 35627 | 47086 | 75.66 | 493 | 100.0 | 13147000000 | True |
| totalshareholderequity | numeric | currency | Total shareholders' equity. | kept | 35201 | 47086 | 74.76 | 493 | 100.0 | 10599000000 | True |
| treasurystock | numeric | currency | Treasury stock (repurchased shares). | kept | 19051 | 47086 | 40.46 | 443 | 89.86 | 0 | True |
