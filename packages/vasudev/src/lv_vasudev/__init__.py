from lv_datasets import Dataset

from lv_vasudev import audit, harmonise, mapping

dataset = Dataset(
    name="vasudev",
    label="Vasudev cohort",
    flow_steps=harmonise.FLOW_STEPS,
    harmonise=harmonise.harmonise,
    audit=audit.run,
    mapping_rows=mapping.mapping_rows,
)

__all__ = ["dataset"]
