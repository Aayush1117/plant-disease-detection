import os

TRAIN_DIR = r"C:\PlantData\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
VALID_DIR = r"C:\PlantData\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"

print("Checking dataset...\n")

train_classes = sorted(os.listdir(TRAIN_DIR))
valid_classes = sorted(os.listdir(VALID_DIR))

print("Training classes:", len(train_classes))
print("Validation classes:", len(valid_classes))

print("\nFirst 10 classes:")
for cls in train_classes[:10]:
    print(cls)

print("\nDataset check completed successfully!")