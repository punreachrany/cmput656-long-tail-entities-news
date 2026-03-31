from refined.inference.processor import Refined
refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                  entity_set="wikipedia")
spans = refined.process_text("Albert Einstein")

print(spans)