# Project Structure

Professional organization of the Visa Processing Days Prediction project.

```
visa/
├── src/                      # Source code
│   ├── api.py               # Flask/FastAPI application
│   └── predict_processing_days.py
│
├── data/                     # Data files
│   ├── visa_dataset.csv
│   ├── visa_dataset.pdf
│   ├── dataset_tracking.json
│   └── preprocessing_info.pkl
│
├── models/                   # Trained ML models
│   └── visa_processing_model.pkl
│
├── notebooks/                # Jupyter notebooks and analysis
│   ├── Milestone1.ipynb
│   ├── MileStone1ProssessingDays.py
│   ├── MileStone2EDAandFE.py
│   ├── Milestone3.py
│   ├── Milestone4.py
│   └── __init__.py
│
├── frontend/                 # Web interface
│   ├── static/              # CSS, JS, images
│   │   ├── css/
│   │   └── js/
│   ├── templates/           # HTML templates
│   ├── index.html
│   ├── config.html
│   └── vercel.json
│
├── tests/                    # Test suite
│   └── test_prediction.py
│
├── config/                   # Configuration files
│   ├── requirements.txt      # Python dependencies
│   ├── runtime.txt          # Python version
│   ├── pip.conf
│   └── apt.txt              # System dependencies
│
├── README.md                 # Project documentation
├── DEBUGGING_GUIDE.md        # Debugging instructions
├── INTEGRATION_STATUS.md     # Integration status
├── LICENSE                   # License file
│
├── Procfile                  # Heroku deployment
├── Procfile.backend          # Backend-specific Procfile
├── railway.toml              # Railway deployment config
├── render.yaml               # Render deployment config
├── nixpacks.toml             # Nix deployment config
│
└── .git/                     # Version control
```

## Directory Purposes

- **src/** - Python source code (API, prediction logic)
- **data/** - Raw and processed data files, tracking info
- **models/** - Trained machine learning models
- **notebooks/** - Jupyter notebooks, data exploration, milestones
- **frontend/** - Web UI (HTML, CSS, JavaScript)
- **tests/** - Unit and integration tests
- **config/** - Dependencies and configuration
- **Root** - Documentation, deployment configs, .gitignore

