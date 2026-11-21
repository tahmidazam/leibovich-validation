from lv_datasets import Dataset

from lv_transorce import audit, harmonise, mapping

dataset = Dataset(
    name="transorce",
    label="TranSORCE",
    flow_steps=harmonise.FLOW_STEPS,
    harmonise=harmonise.harmonise,
    audit=audit.run,
    mapping_rows=mapping.mapping_rows,
    mapping_tables=mapping.mapping_tables,
)

__all__ = ["dataset"]
