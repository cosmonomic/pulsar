from fastapi import Depends
from typing import Annotated
from fastapi.requests import Request
from pulsar.rt import Pulsar as _Pulsar


def get_pulsar(request: Request) -> _Pulsar:
    return request.app.state.pulsar


type Pulsar = Annotated[_Pulsar, Depends(get_pulsar)]
