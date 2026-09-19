# Part 1: Problem Definition and Data Collection

## Problem definition

Buying a house is one of the largest financial decisions most people make, and the price a
property eventually sells for depends on a mix of factors that are not always obvious from a
listing alone — location, land size, bedroom/bathroom count, and the broader dynamics of the
suburb it sits in. This project frames property valuation as a supervised regression problem:
given a set of property characteristics, predict the sale price, and use that model to support
(not replace) a buyer's decision-making. Acting as a data scientist for a real estate agency, the
goal is not simply to chase the lowest prediction error but to make and justify a series of
defensible modelling decisions, and to be honest about where the resulting model can and
cannot be trusted.

## Suburb selection and rationale

Three Sydney suburbs were selected to represent meaningfully different segments of the
housing market: **Kellyville** (postcode 2155), **Cherrybrook** (postcode 2126), and
**Blacktown** (postcode 2148). All three sit in Sydney's north-west/western growth corridor,
which keeps some factors (state, general region, access to the same rail corridors) roughly
comparable, while still varying enough on price, density and demographics to make the
prediction problem non-trivial.

- **Kellyville** is a rapidly-growing Hills District suburb around 36 km north-west of the
  Sydney CBD, served by the Sydney Metro Northwest. It is dominated by newer, larger
  four-to-five-bedroom family homes, with a median house price in the order of $1.8M and a
  relatively young, family-oriented population.
- **Cherrybrook** sits about 27 km north-west of the CBD in the Hornsby Shire and is one of
  the more affluent, established suburbs in the sample — leafy, low-density, an older median
  age, and a median house price in the $2.4–2.6M range, noticeably above Kellyville despite
  being closer to the CBD.
- **Blacktown** is roughly 34 km west of the CBD and is the largest and most demographically
  diverse of the three, with a younger population, a higher proportion of renters, greater
  housing density, and a substantially lower median house price (roughly $0.9–1.1M).

Together these suburbs vary on the features expected to matter most for price — distance to
the CBD, land size and dwelling age, and suburb-level socio-economic profile — which should
let the models pick up genuine price drivers rather than just memorising one narrow market.

## Data collection

Sold-property data was manually collected from realestate.com.au and domain.com.au sold-listing
search results for each suburb, recording, for every property: address, suburb, postcode,
bedrooms, bathrooms, car spaces, land size (sqm), sale price, and sale date. 100 Kellyville and
100 Cherrybrook properties were collected this way. For Blacktown, 32 listings were sourced
across Domain and individual agency sites (e.g. Raine & Horne), and the schema was extended to
also capture property type, sale method, distance to CBD, an agent-description snippet, and the
source listing URL, with a view to using the richer fields (e.g. agent text) as engineered
features later in the project. 232 sold properties were collected in total, comfortably above
the assignment's 100-property/30-per-suburb minimum.

## Data quality, challenges, and limitations

Several data quality issues emerged during collection and are worth being upfront about:

- **Data entry errors, caught and corrected.** While expanding the Blacktown sample, three
  existing rows (16 Timmins Walkway, 10 Western Crescent, 18 Coolabah Place) turned out to have
  had their sale prices scrambled between addresses in the original collection pass. This was
  only caught because the source listings were re-checked; it is a reminder that manually
  collected data needs spot-verification, not just a plausibility check.
- **Schema inconsistency.** The richer Blacktown fields (land size, distance to CBD, agent
  description) are largely unfilled — most source listing pages simply did not state land size,
  and distance-to-CBD/agent-description text was not collected for any suburb — so these fields
  are not yet usable as predictors without further backfilling.
- **Temporal inconsistency.** Kellyville and Cherrybrook sales are all from April–September
  2026, while Blacktown sales span November 2023 to July 2026. Since prices generally drift
  upward over time, comparing a suburb's prices across a three-year window against two suburbs
  sampled in a single half-year risks confounding "suburb effect" with "time effect," and will
  be handled explicitly in the EDA (e.g. by checking for a price/date trend within suburb).
- **Incomplete sale dates.** Three of the newly-added Blacktown properties (17 Cardiff Street,
  263A Flushcombe Road, 37 Charles Street) did not have a sale date stated on their source
  listing page; these are flagged rather than guessed, and will need to be either excluded or
  looked up before being used in any date-sensitive analysis.
- **Collection bias.** Because the dataset was built manually from sold-listing search pages,
  it likely over-represents properties that were actively marketed and successfully listed
  online (rather than off-market sales), and skews toward standalone houses — two of the new
  Blacktown rows are townhouses, flagged separately, and would need to be excluded or modelled
  with a property-type feature. This limits how confidently the resulting model generalises to
  the full housing stock of each suburb.
- **No independent validation beyond the checks above.** Figures were taken as reported by the
  listing platforms; further undetected data-entry errors on the source sites cannot be fully
  ruled out.

These limitations are treated as a starting point for the EDA in Part 2, not a reason to discard
the data — but they do mean price comparisons across suburbs, and any features built from the
Blacktown-only columns, need to be interpreted cautiously until the sample is complete and
consistent.

---
*Sources for suburb context: OpenAgent, Domain, and GDP.com.au suburb profiles (Kellyville,
Cherrybrook, Blacktown), accessed September 2026.*
