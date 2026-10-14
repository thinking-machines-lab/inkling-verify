FROM docker.io/library/python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

# Support rootless builders with a single mapped UID.
RUN mkdir -p /usr/local/share/fonts \
    && apt-get -o APT::Sandbox::User=root update \
    && apt-get -o APT::Sandbox::User=root install -y --no-install-recommends tesseract-ocr libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --only-binary=:all: \
    pillow==12.2.0 numpy==2.4.6 matplotlib==3.10.9 scipy==1.17.1 \
    opencv-python-headless==4.13.0.92 scikit-image==0.26.0 sympy==1.14.0 \
    scikit-learn==1.9.0 shapely==2.1.2 pandas==3.0.3 pytesseract==0.3.13
RUN mkdir -p /sandbox/images /tmp/matplotlib
COPY _python_kernel.py /opt/python_kernel.py
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /sandbox
