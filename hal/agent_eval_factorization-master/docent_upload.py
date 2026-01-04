import importlib
import os
from compile_traces import trace_paths_from_dir

docent_module = importlib.import_module("hal-paper-analysis.qualitative.full_pipeline")


# You can then access its contents:
# my_module.some_function()
# my_module.some_variable
all_results = []


# connect to collection
collection_id = "36c05561-71f8-4677-9ca5-0c1a5da436aa"
docent_uploader = docent_module.DocentUploader(api_key=os.getenv("DOCENT_API_KEY", "dk_CrDnMTOkAP43XyWL_PCJts8Wic0HXx7rn8ZDO07gP026W8FOOygBzz0NF5penxK"))

# load traces
trace_directories = trace_paths_from_dir("/traces")

# convert traces
docent_results, conversion_stats = docent_module.DocentConverter.convert_to_docent_messages(all_results)
print(f"✅ Conversion complete!")
print("📊 Conversion stats:")
print(f"   Total messages: {conversion_stats['total_messages']}")
print(f"   Successfully converted: {conversion_stats['successful']}")
print(f"   Failed to convert: {conversion_stats['failed']}")
if conversion_stats['total_messages'] > 0:
    success_rate = conversion_stats['successful']/max(1, conversion_stats['total_messages'])*100
    print(f"   Success rate: {success_rate:.1f}%")

# upload results to docent
print(f"\n📤 Uploading to docent collection '{collection_id}'...")
collection_id, upload_stats = docent_module.upload_to_docent(docent_uploader, docent_results, collection_id, batch_by_model=True)