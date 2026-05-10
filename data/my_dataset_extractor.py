import os, json

SAHTE_DIR = r'C:\Users\Tugba Gul\Desktop\sahte_veriseti'

samples = []

for i in range(1, 11):
    # Sahte (output_1.jpg ... output_10.jpg)
    fake_path = os.path.join(SAHTE_DIR, f'output_{i}.jpg')
    if os.path.exists(fake_path):
        samples.append({'img': fake_path, 'label': 1})

    # Gerçek (sahte_1.jpg ... sahte_10.jpg — manipüle edilmemiş orijinaller)
    real_path = os.path.join(SAHTE_DIR, f'sahte_{i}.jpg')
    if os.path.exists(real_path):
        samples.append({'img': real_path, 'label': 0})

# tugba.jpg de gerçek
tugba_path = os.path.join(SAHTE_DIR, 'tugba.jpg')
if os.path.exists(tugba_path):
    samples.append({'img': tugba_path, 'label': 0})

output_path = r'C:\Users\Tugba Gul\Desktop\faceforencis\my_dataset.json'
with open(output_path, 'w') as f:
    json.dump(samples, f, indent=2)

print(f"Toplam örnek: {len(samples)}")
print(f"Gerçek: {sum(1 for s in samples if s['label'] == 0)}")
print(f"Sahte: {sum(1 for s in samples if s['label'] == 1)}")
print(f"Kaydedildi: {output_path}")
