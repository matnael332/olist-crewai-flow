# workforce_saas: the synthetic data

`synth.py` invents the raw layer of `SCHEMA.md`: thirteen tables that look like a time-attendance
SaaS exported them, generated from a causal story so that a model has real signals to learn and a
hidden truth to be checked against.

```bash
uv run synth --data-dir data                       # 200 accounts, 2022-01 .. 2025-08, ~145 MB
uv run synth --accounts 300 --seed 7 --force        # bigger, another world, overwrite
uv run synth --data-dir /tmp/t --accounts 25 --start-month 2023-01 --snapshot 2024-07-01 --dirt 0
```

Output under `<data-dir>`:

| path | content |
|---|---|
| `raw/<table>.csv`, `raw/raw_schema.json`, `raw/manifest.json` | what the pipeline reads (`--data-dir`) |
| `truth/accounts_truth.csv` | per account: archetype, initial headcount, decline start, trouble start, cancellation and its reason |
| `truth/monthly_health.csv` | per account-month: latent health, hazard, scheduling on, `churned_60d` |
| `truth/dirt_counts.json` | how many rows each dirt kind touched, per table |
| `lookups/` | empty; the spec needs no lookups |

Two runs with the same config are byte-identical. Every account draws from its own generator
(`seed`, account index), so `--accounts 300` reproduces the first 200 accounts of the default
dataset exactly and appends 100 more.

## The story of an account

An account signs up on a random day (trial of 14 or 30 days), belongs to an industry and a
region, and follows one of eight **archetypes**. Each archetype has a latent monthly **health**
`h` in [0, 1] and every table is a function of it:

| archetype | share | health path | how it ends |
|---|---|---|---|
| steady | 34 % | 0.85 with mild autocorrelated noise | random hazard only |
| growing | 12 % | 0.80 rising 0.01 a month; headcount +4 % a month | random hazard |
| shrinking | 8 % | 0.80 falling 0.02 a month after month 6; headcount -3 % a month; one site closes | random hazard, higher |
| seasonal | 8 % | 0.80 plus a yearly sine of 0.15 (peak in July) | random hazard |
| trial_bounce | 14 % | 0.40 during the trial, never converts | cancels 0-14 days after the trial (`trial_not_converted`) |
| slow_fade | 12 % | 0.85 until a random month, then down to 0.15 over 4-6 months | hazard rises with the fade; forced cancel one month after it ends (`fade`) |
| price_sensitive | 7 % | 0.75; from a random month pays later every month, last invoice unpaid | cancels 2-4 months after the first late payment (`price`) |
| device_trouble | 5 % | 0.80; one clock goes offline for 1-3 streaks of 10-40 days | random hazard only: a dead clock is not a closed site |

**Hazard.** After the trial, the monthly cancellation probability is
`sigmoid(alpha + 6 (1 - h))` with `alpha` -6.0 (steady, seasonal, device_trouble), -6.5 (growing),
-5.0 (shrinking), -5.5 (slow_fade, price_sensitive) and -4.0 while a price-sensitive account pays
late. Over the default dataset this gives roughly 2 % monthly churn among converted accounts and
a `churned_60d` positive rate of about 5 % at account-month grain.

**Lead signals.** Because every signal is a function of `h`, a fading account shows them in a
fixed order, which is the order `SCHEMA.md` describes:

| months before the cancellation | what changes | formula |
|---|---|---|
| 4-6 | admin logins thin out | `Poisson(rate · h²)` per business day |
| 2-3 | scheduling module switched off, schedule published later | on while `h >= 0.5`; lead days mean `2 + 8h` |
| 2-3 | punches decline, more of them edited | attendance `× min(1, 1.3h)`; edited share `0.03 + 0.12 (1 - h)` |
| 1-2 | payroll export skipped, more corrections | skipped with `(1 - h)²`; corrections `Poisson(0.5 + 3 (1 - h) + 0.02 n)` |
| 1-2 | employees leave faster, requests wait longer | leave rate `0.025 + 0.08 (1 - h)`; decision delay mean `1 + 6 (1 - h)` days |

## The tables

- **Organisation.** Initial headcount lognormal around an industry median (5-12, capped at 150),
  1-4 sites, roles staff / shift_lead / manager, employment types full_time / part_time /
  contractor. Hourly wages 34 ILS times a lognormal factor and a role multiplier, with occasional
  anniversary raises.
- **Daily engine.** One employees-by-days grid per account. Work week by industry (restaurants
  and retail seven days, construction Sunday to Friday, the rest Sunday to Thursday), attendance
  probability by employment type, a usual shift start per employee, shift length about 8.3 hours
  (8.8 in manufacturing), stretched when health is low.
- **Punches** are the in/out pair of every attendance day. At a site with a clock, 80 % use it
  (biometric or nfc); otherwise the app (80 %) or the web. During a device streak, punches at that
  site drop to 58 % and half of them move to the app.
- **Hours** are derived from the pair: regular up to 8.6 h, 125 % up to two more, 150 % beyond,
  night hours between 22:00 and 06:00, a rest violation above 12 h or under 8 h since the previous
  shift.
- **Shifts** exist only while the scheduling module is on: one planned shift per attendance day
  (`worked`, `late` beyond 15 minutes, 3 % `swapped`), plus `no_show` and unassigned `open`
  shifts whose rates rise as health falls.
- **Absence requests** while the module is on: a quarter of a request per employee and month,
  sick / vacation / unpaid / other, 90 % decided, 15 % of those rejected.
- **Payroll exports** once per period a few days into the next month, by one of five payroll
  packages per account.
- **Usage** every day: admin logins, reports, module flags, messages, app installs (employees who
  use the app and are still employed), support tickets (spiking during device streaks and late
  payments).
- **Devices** at 55 % of accounts (always for device_trouble): one per site, two for sites above
  25 people, 10 % replaced once. **Device health** per device and day: mostly online, exponential
  offline hours otherwise, near-total outage during a streak, failed recognitions rising just
  before it.
- **Billing** one row per calendar month from the sign-up month to the cancellation month: seats
  are the employees at the period start, the plan follows the seats (basic under 8, standard to
  25, premium above; 18 / 28 / 42 ILS a seat), trial months bill nothing, payment 14 days after
  the period plus the lateness (0 for 70 % of invoices, geometric otherwise), empty when unpaid or
  not yet due at the snapshot.
- **Accounts**: `status` is `cancelled`, `trial` (trial still running at the snapshot) or `active`;
  `cancelled_at` is set only for cancelled accounts.

Events stop at the cancellation or one second before the snapshot. Accounts whose cancellation
would fall after the snapshot are simply active: the right censoring that the clean layer
handles is real in the data.

## Dirt

`--dirt f` (default 0.02) plants realistic imperfections after the ids are assigned, each counted
in `truth/dirt_counts.json`. Headers, keys other than exact duplicates, `account_id` and the
timestamp format are never touched, so `load_raw_tables` always succeeds.

| kind | tables | share | effect |
|---|---|---|---|
| duplicate_rows | punches, hours_daily, absence_requests | f / 2 | exact copies |
| missing_out | punches | f of pairs | the `out` punch is missing |
| null_non_key | role, employment_type, method, device_type, payroll_software, kind | f | empty value |
| bad_enum | method, outcome, kind | f / 4 | `App`, `mobile`, `WORKED`, `Sick` |
| swapped_range | absence start/end, shift planned start/end | f / 4 | the two values swapped |
| late_edit | punches | f | an extra corrected copy, 5-40 minutes away, `edited` by an admin |

## Notes for the clean step (step 4)

- `app_install_share = app_installs_total / active_employees` can exceed 1 (installs count every
  employed app user, active employees only those who punched); clip to [0, 1].
- `payroll_exported` for month M means "period M-1 was exported by the end of M": join exports on
  `period == M - 1` and `exported_at <= month_end`.
- A site is active only through its employees' punches, so `active_employees >= active_sites`
  (rule C03) holds by construction.
- The truth files are for checking the model, never for training: the pipeline must not read
  `truth/`.
- `clean.py` implements these notes; `tests/test_workforce_spec.py` checks the clean table and the
  model against the truth files.
