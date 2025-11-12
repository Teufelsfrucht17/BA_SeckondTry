# DAX Momentum Forecasting

Dieses Projekt implementiert eine momentum-basierte Vorhersagepipeline für ausgewählte DAX-Titel im 30-Minuten-Raster. Das Modell sagt den erwarteten nächsten 30-Minuten-Return voraus und leitet daraus LONG/SHORT-Signale ab.

## Projektüberblick

* **Zieluniversum:** 6 DAX-Aktien (konfigurierbar via `config.yaml`).
* **Zeitraster:** 30-Minuten-Bars.
* **Modell:** LSTM-Regressor in PyTorch.
* **Validierung:** Zeitreihenbewusst (Train/Test-Split + `TimeSeriesSplit`).
* **Datenquelle:** Refinitiv Workspace / LSEG Data Library (mit Offline-Mock für lokale Tests).

Die Architektur trennt Datenzugriff, Feature Engineering, Modellierung und ausführbare Pipelines sauber voneinander.

## Setup

1. **Virtuelle Umgebung anlegen**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```

2. **Abhängigkeiten installieren**

   ```bash
   pip install -r requirements.txt
   ```

3. **Umgebungsvariablen setzen**

   *Kopiere* `.env.example` zu `.env` und trage deine Refinitiv-Zugangsdaten ein (z. B. `APP_KEY`).
   Lade die Variablen beispielsweise mit [`python-dotenv`](https://pypi.org/project/python-dotenv/) oder exportiere sie direkt in deiner Shell.

## Konfiguration

Alle Pipelineparameter liegen in `config.yaml` und umfassen Ticker, Zeitintervall, Featureauswahl, Modellhyperparameter, Pfade und Live-Stub-Einstellungen. Passe sie nach Bedarf an.

## Datenabruf

```bash
python -m src.data.fetch_history --config config.yaml
```

* Ruft historische OHLCV-Daten über Refinitiv ab (oder generiert Mock-Daten bei fehlender Verbindung).
* Speichert die Historie als `artifacts/history.parquet`.

## Training

```bash
python -m src.pipeline.run_train --config config.yaml
```

* Lädt Historie und Features.
* Skalierung, Sequenzbildung, Cross-Validation mit `TimeSeriesSplit`.
* Trainiert ein LSTM, speichert bestes Modell (`models/best_model.pt`) sowie Scaler (`artifacts/scaler.pkl`).
* Berichtigt Trainings- und Validierungsmetriken (MSE/R²).

## Signal-Backtest (Stub)

```bash
python -m src.pipeline.run_backtest_signals --config config.yaml
```

* Nutzt gespeichertes Modell, erzeugt Vorhersagen und LONG/SHORT/FLAT-Signale.
* Berechnet Signalverteilung und Trefferquote (gegen Vorzeichen der realisierten Returns).
* Exportiert Ergebnisse nach `artifacts/signals.csv`.

## Live-Stub

```bash
python -m src.pipeline.run_live_stub --config config.yaml
```

* Veranschaulicht, wie alle 30 Minuten neue Bars abgerufen und durch die Pipeline geschickt werden.
* Nutzt einen Rolling-Window-Puffer je RIC und gibt das Handelssignal aus.
* Enthält TODO-Hinweise für die produktive Refinitiv-Integration.

## Tests

Unit-Tests für Feature-Berechnung, Sequenzierung und Signallogik:

```bash
pytest
```

## Erweiterungen & TODOs

* Weitere Features (z. B. gleitende Durchschnitte, RSI, MACD, Volatilitätsmaße).
* Portfolio-Aggregation, Risikoparameter und echtes PnL-Backtesting.
* Deployment der Live-Pipeline (Scheduler, Persistenz der Rolling-Window-States).

## Hinweise

* Train-/Live-Pipelines teilen sich denselben Feature- und Skalierungsprozess, um Drift zu vermeiden.
* Overfitting-Monitoring: Vergleiche Training- und Out-of-Sample-R².
* Stelle sicher, dass deine Refinitiv-Session aktiv ist, falls du echte Daten abrufen möchtest.
