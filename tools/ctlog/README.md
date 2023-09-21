### Download Certificate Transparency logs from [crt.sh](https://crt.sh/) 

1. Install Docker and Docker compose  
```sudo apt install docker-compose-plugin```

2. Run docker container  
```docker-compose up -d```

### Output newly observed domain names within 24 hours

1. Install python packages  
```pip install -r requirements.txt```

2. Output domain names to data/yyyymmdd.txt (set job scheduler if you want to run this script periodically)  
```python3 get_domain.py```  
