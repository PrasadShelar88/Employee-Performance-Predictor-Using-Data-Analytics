from app.data_generator import save_dataset

if __name__ == "__main__":
    df = save_dataset()
    print(f"Dataset generated with {len(df)} rows.")
