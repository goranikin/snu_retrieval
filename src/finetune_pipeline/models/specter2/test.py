from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query")
print("Adapter load success!")
