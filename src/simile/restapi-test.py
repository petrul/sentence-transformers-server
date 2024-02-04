from fastapi import FastAPI
from fastapi.testclient import TestClient

import unittest
from util import *
from encoder import *
from restapi import *

class RestApiTest(unittest.TestCase):

    client = TestClient(app)

    def test_api(self):
        names = self.client.get("/api/models/names")
        
        assert names.status_code == 200
        assert len(names.json()) == 2
        
        response_mini = self.client.post(f"/api/models/{NAME_ALL_MINILM_L6_V2}/encode",
                                    json=['hello', 'world']).json()
        assert len(response_mini) == 2
        # again, response should be the same
        assert response_mini[0] == self.client.post(f"/api/models/{NAME_ALL_MINILM_L6_V2}/encode",
                                    json=['hello']).json()[0]
        
        resp_mpnet = self.client.post(f"/api/models/{NAME_ALL_MPNET_BASE_V2}/encode",
                                    json=['hello', 'world']).json()
        assert len(resp_mpnet) == 2
        assert resp_mpnet[0] != response_mini[0]
        
    def test_info(self):
        info = self.client.get(f"/api/models/{NAME_ALL_MINILM_L6_V2}/info")
        assert info.json()['name'] == NAME_ALL_MINILM_L6_V2
        assert info.json()['max_seq_length'] == 256

if __name__  == '__main__':
    unittest.main()
    # RestApiTest().test_info()