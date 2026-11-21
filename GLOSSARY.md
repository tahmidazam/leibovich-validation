# Glossary

**Dataset**: one source of patients the model is validated on, such as TranSORCE or the Vasudev cohort. It is the unit the app registers, harmonises, scores and reports, and it is named by its registry `name`. Use "dataset" in code, commands and documentation. _Avoid_: cohort. "cohort" survives only inside a published dataset's own name (the Vasudev cohort), in the `cohort` column of the harmonised table, and in older module and function names.

**Adapter**: the package that reads one dataset's raw files and describes it to the app as a `Dataset`. _Avoid_: loader, plugin.

**Dataset descriptor** (`Dataset`): the object an adapter registers. It holds the name, display label, flow steps and the `harmonise` function, and optionally the audit and mapping hooks.

**Registry**: the lookup from dataset name to `Dataset`, filled from the `leibovich_validation.datasets` entry-point group. Tests replace it with `lv_datasets.override`.

**Harmonised table**: the per-patient table in the columns of `lv_datasets.HARMONISED_COLUMNS`, the same for every dataset.

**Excluded column**: the `excluded` column of what `harmonise` returns. Null for a retained patient, otherwise the name of the first exclusion step that removed the row.

**Flow step**: one named line of the flow table: a main count, an exclusion or a note. Each dataset declares its own in `flow_steps`.

**Analysis set**: the patients left after every exclusion, the rows of the harmonised patient table.
