import torch
from transformers import LlamaConfig

class KV_Cache:

    def __init__(self, 
        config :LlamaConfig,
        batch_size :int = 1,
        max_length :int = 256, 
        device :str = 'cuda:0',
        dtype = torch.float16) -> None:
        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        """KV_Cache 객체가 처음 생성될 때, key와 value vector를 저장할 비어있는 메모리 공간(텐서)을 미리 할당하고 준비함
        > k_cache와 v_cache라는 두 개의 거대한 텐서를 모두 0으로 채워 생성
        
        """

        # ***KV에서 k에 대한 shape과 initialization***
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        # ***KV에서 v에 대한 shape과 initialization***
        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0

    def initialize_kv(self,
            k_cache :torch.Tensor,
            v_cache :torch.Tensor,
            kv_len :int):
        """이미 존재하는 KV cache값(예; 다른곳에서 계산된 프롬프트의 KV캐시 등)을 가져와 현재 객체의 초기 상태로 설정
        > 외부에서 전달 받은 k_cache와 v_cache 텐서를 현재 객체의 self.k_cache와 self.v_cache에 복사

        """
        
        self.k_cache[...,:kv_len,:] = k_cache[...,:kv_len,:]
        self.v_cache[...,:kv_len,:] = v_cache[...,:kv_len,:]

        self.kv_offset = kv_len
        
        
    
    def gather_kv(self, indices: list[int]):
    
        self.k_cache[..., :len(indices), :] = self.k_cache[..., indices, :]
        self.v_cache[..., :len(indices), :] = self.v_cache[..., indices, :]

        self.k_cache[..., len(indices):, :] = 0.0
        self.v_cache[..., len(indices):, :] = 0.0

        self.kv_offset = len(indices)
    
    def gather_kv_incremental(self, indices: list[int], offset:int):
    """SD의 검증 단계가 끝난 후, 수락된 토큰 경로에 맞춰 KV cache를 효율적으로 재정렬하는 가장 핵심적인 one of the methods
    > SpecTree.py의 .verify() method에서 호출되며, 수락된 토큰들의 indices를 받음
    > Indices를 주소로 삼아, 흩어져 있는 유효한 KV cache 값들만 골라내어 텐서의 앞부분 부터 빈틈없이 다시 채움
    > 이 과정을 통해 버려진 branch들의 cache는 삭제되고, 다음 스텝이 필요한 memory만 남음

    """
        self.k_cache[..., offset:offset + len(indices), :] = self.k_cache[..., indices, :]
        self.v_cache[..., offset:offset + len(indices), :] = self.v_cache[..., indices, :]

        self.k_cache[..., offset + len(indices):, :] = 0.0
        self.v_cache[..., offset + len(indices):, :] = 0.0

        self.kv_offset = offset + len(indices)


    
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            storage_ids :torch.LongTensor,
            debug :bool = False):
        """모델의 forward pass 중에 새롭게 계산된 Key와 Value vector를 cache에 추가
        > Llama_modules.py의 attention 계산 부분에서 호출
        > new_k_cache, new_v_cache를 storage_ids가 알려주는 정확한 위치에 index_copy_를 이용해 기록

        """
        
        input_length = len(storage_ids)
        if debug:
            assert input_length == new_k_cache.shape[-2]
            assert input_length == new_v_cache.shape[-2]
        
        self.k_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_k_cache)
        self.v_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_v_cache)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += input_length
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0
    
    def get_usable_length(self, layer_idx:int, input_length :int):
            if layer_idx == self.num_layers - 1:
                return self.kv_offset
            else:
                return self.kv_offset + input_length
    
    def set_kv_len(self, kv_len :int):
            self.kv_offset = kv_len
    
        
