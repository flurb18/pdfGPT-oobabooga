FROM atinoda/text-generation-webui:default
COPY ./requirements.txt /requirements.txt
RUN pip install /requirements.txt
RUN mkdir /app/extensions/pdfGPT_oobabooga
COPY ./script.py /app/extensions/pdfGPT_oobabooga/script.py