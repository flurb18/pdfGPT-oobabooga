FROM atinoda/text-generation-webui:default
COPY ./requirements.txt /requirements.txt
RUN pip install /requirements.txt
RUN mkdir /app/extensions/pdfGPT-oobabooga
COPY ./script.py /app/extensions/pdfGPT-oobabooga/script.py