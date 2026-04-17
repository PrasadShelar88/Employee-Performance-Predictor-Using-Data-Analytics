from app.ml import train_and_save

if __name__ == "__main__":
    metrics = train_and_save()
    print("Training complete")
    print(metrics)
