FROM python:3.11-bookworm as requirements-stage
WORKDIR /tmp

RUN pip install poetry

COPY ./pyproject.toml /tmp/

RUN poetry lock

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.11-bookworm
WORKDIR /workspace

COPY --from=requirements-stage /tmp/requirements.txt /workspace/requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \ 
    pip install --no-cache-dir --upgrade -r /workspace/requirements.txt

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--IdentityProvider.token=''"]