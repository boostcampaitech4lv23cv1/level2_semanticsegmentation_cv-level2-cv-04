# How to start
```
pip install streamlit
```
protobuf 관련 오류가 난다면:
```
pip install --upgrade protobuf
```
# run

현재 폴더에 output.csv 넣거나 parser 인자로 준 후
```
streamlit run app.py --server.fileWatcherType=none --server.port=30005
```