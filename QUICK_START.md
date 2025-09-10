# 🚀 QUICK START GUIDE - PII Redactor Pro

## 🎯 Modern Setup (2 minutes)

### Super Quick Setup

```bash
# One command does everything!
python setup.py
```

### Using Make (if you have it)

```bash
make setup
```

### What happens automatically:

✅ Installs Poetry (modern Python dependency management)  
✅ Creates virtual environment  
✅ Installs all dependencies  
✅ Sets up your Azure credentials from `.env.template`  
✅ Offers to launch the app immediately

## 🔄 Running the App

### Using Make (Recommended)

```bash
make run
```

### Using Poetry

```bash
poetry run python pii_redactor_app.py
```

### All Available Commands

```bash
make help          # See all commands
make dev           # Development setup
make test          # Run tests
make format        # Format code
make clean         # Clean up files
```

## 4️⃣ Process Your First File

1. Click **"📁 Upload CSV File"**
2. Select your Zendesk export CSV
3. Click **"👁 Preview First Column"** to test
4. Click **"🚀 Start Redaction"** to process entire file
5. Find redacted file in same folder as original

## 5️⃣ Batch Processing (Multiple Files)

```bash
# Process all CSV files in a directory
python batch_process.py --dir /path/to/csv/files

# Use local detection (no Azure required)
python batch_process.py --local
```

## 📝 Test with Sample Data

A sample Zendesk CSV file is included: `sample_zendesk_data.csv`

## 🆘 Troubleshooting

### Azure Connection Issues

- Verify endpoint includes `https://` and ends with `/`
- Check API key is correct (no extra spaces)
- Ensure internet connection is active

### Missing Dependencies

```bash
pip install customtkinter pandas azure-ai-textanalytics python-dotenv
```

### Permission Errors

```bash
# On Mac/Linux, make scripts executable
chmod +x setup.py batch_process.py
```

## 🎯 Key Features

- ✅ **100+ PII Types** detected and redacted
- ✅ **Real-time Progress** with ETA
- ✅ **Cost Tracking** per run and total
- ✅ **Preview Mode** to test before full processing
- ✅ **History Tracking** of all processed files
- ✅ **Batch Processing** for multiple files
- ✅ **Dark/Light Theme** support

## 💰 Pricing

- Azure Text Analytics: ~$1 per 1 million characters
- Average CSV (1000 rows × 10 columns): ~$0.05-0.10
- Cost displayed in real-time during processing

## 🔒 Security

- Credentials stored locally in `.env`
- No data retained by Azure after processing
- All communication over HTTPS
- Processing history stored locally only

---

**Ready to redact!** Launch the app and start protecting PII in your data. 🛡️
