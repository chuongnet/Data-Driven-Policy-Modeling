from CCDGeneration import CCDGeneration

filepath = "../data/entities.json"    # for window OS, "..\data\entity.json"
# filepath = "../data/efficient.json"
# filepath = "../data/largefile.json"
filename = 'temp'
try:
    gen = CCDGeneration(filepath)
    print(gen.write_file(filename))
except Exception as e:
    print(f'Error: {e.args[0]}')
