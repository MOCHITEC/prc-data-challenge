# Stamp rates (2026-10-07): not adopted
14 leakage-free, EB-shrunk "schedule-copy" rate columns (airline → airline×ADES → flight number, month-out).
- LIRF matched: +44 MSE [28, 61], but the top-50 share is 0.89, and the gain turns negative when the flight number
  is unseen (48 % of 2026 July LIRF flight numbers are unseen: rotating Ryanair/Wizz callsigns).
- Other airports' matched lanes and the unmatched lane: no signal, or negative.
Script: `analysis/local_stamp_rates.py`.
