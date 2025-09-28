# Climate Sim Backend

Backend service for scheduling and running climate simulations with a C++ engine.  
Built with **FastAPI**, **Celery**, and **Redis**.

## Quickstart

### 1. Clone repo
```bash
git clone https://github.com/antoniorizzoeng/climate-sim-backend.git
cd climate-sim-backend
```
### Run with Docker Compose
```bash
docker-compose up --build
```

### Submit a simulation
```bash
curl -X POST http://HOST/simulate
```