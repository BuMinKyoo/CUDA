###### Top

  - [프로젝트](#프로젝트)
  - [CUDA기초](#CUDA기초)


<br/>
<br/>

***

#프로젝트
  - [CudaPrectice](https://github.com/BuMinKyoo/CudaPrectice)


<br/>
<br/>

***

#CUDA기초

  - __global__ :	실행되는곳 -> GPU / 호출하는 곳 -> CPU / (또는 GPU, 동적 병렬 처리)	반드시 void, <<<>>>로 호출, 비동기
  - __device__ : 실행되는곳 -> GPU	/ 호출하는 곳 -> GPU	/ GPU 안에서만 쓰는 보조 함수
  - __host__ : 실행되는곳 -> CPU	/ 호출하는 곳 ->CPU / 기본값이라 보통 생략
  - _host__ __device__를 같이 붙이면 CPU/GPU 양쪽에서 쓸 수 있는 함수

<br/>

~~~C++
__global__ void WriteIndex(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}
~~~

<br/>
<br/>

  - "층"은 설치 단위가 아니라 호출 관계임. 위층이 아래층을 부를 뿐이고, 중간을 건너뛰어도 됨. 그리고 맨 아래 커널 드라이버를 빼면 전부 내 프로세스 안에서 도는 라이브러리임
    - 상위층 : cuBLAS,cuDNN,cuFFT,Thrust 선택적 상위 라이브러리 (여기엔 없음)
    - 중간층 : CUDA Runtime API  ← cudaMalloc, cudaMemcpy, cudaGetDeviceProperties …
      - 헤더: cuda_runtime.h  /  라이브러리: cudart
    - 하위층 : CUDA Driver API   ← cuMemAlloc, cuLaunchKernel … (접두사가 cu, 저수준)
      - 헤더: cuda.h  /  nvcuda.dll

<br/>

```text
┌──────────────────────────────────────────────────────────┐
│                  내 프로세스 (사용자 모드)                 │
│                                                          │
│   내 코드 (main.cu)              ★ 우리가 있는 곳         │
│        │                                                 │
│        │      [상위층] cuBLAS · cuFFT · cuRAND            │
│        │               cuDNN · Thrust/CUB                │
│        │               └ 선택적. 안 쓰면 없어도 그만       │
│        ↓                      ↓                          │
│   ┌────────────────────────────────────────┐             │
│   │ [중간층] CUDA Runtime API               │            │
│   │   cudaMalloc, cudaMemcpy, <<<>>>        │ ← 우리가 쓰는 것│
│   │   cudart_static.lib (exe 안에 포함)      │           │
│   └────────────────────────────────────────┘             │
│        ↓                                                 │
│   ┌────────────────────────────────────────┐             │
│   │ [하위층] CUDA Driver API                 │            │
│   │   cuMemAlloc, cuLaunchKernel            │            │
│   │   nvcuda.dll (드라이버가 설치)            │           │
│   └────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────┘
         ↓  ← 여기서만 OS 커널로 진입
    nvlddmkm.sys (커널 모드 드라이버) + Windows WDDM
         ↓
       [ GPU ]
```

<br/>

  - 층별 비교

| | 상위층 | 중간층 (Runtime) | 하위층 (Driver) |
|---|---|---|---|
| 무엇 | 이미 만들어진 알고리즘 | 쓰기 쉬운 CUDA 기본기 | 날것의 저수준 제어 |
| 예 | `cublasSgemm`, `cufftExecC2C` | `cudaMalloc`, `cudaMemcpy` | `cuMemAlloc`, `cuLaunchKernel` |
| 접두사 | 라이브러리마다 다름 | `cuda~` | `cu~` |
| 헤더 | `cublas_v2.h` 등 | `cuda_runtime.h` | `cuda.h` |
| 실제 파일 | `cublas.lib` + DLL | `cudart_static.lib` | `nvcuda.dll` |
| 설치 주체 | 툴킷 (cuDNN만 별도) | 툴킷 | 그래픽 드라이버 |
| 버전 결정 | 빌드할 때 | 빌드할 때 | 드라이버 업데이트로 |
| 이 프로젝트 | 안 씀 | **씀** | 직접은 안 씀 |

<br/>

- 설치는 두 번뿐

```text
그래픽 드라이버 설치  →  nvcuda.dll (하위층)
CUDA Toolkit 설치    →  nvcc + 헤더 + Runtime + cuBLAS/cuFFT/cuRAND/Thrust
                        (cuDNN만 NVIDIA 사이트에서 별도 다운로드)
```

<br/>

- 헷갈리기 쉬운 것
  - Runtime과 Driver는 둘 중 하나만 쓴다. 같은 일을 하는 두 방식이다 ,Runtime이 내부에서 Driver를 부르지만 그건 라이브러리가 알아서 하는 일이다.
  - Driver API의 파일은 두 곳에서 온다,  헤더 `cuda.h`와 stub `cuda.lib`은 툴킷, 실제 구현 `nvcuda.dll`은 그래픽 드라이버
  - cuDNN만 별도 설치이고, PyTorch·TensorFlow가 내부에서 쓴다.

<br/>

- 중간층과, 하위층 헷갈리는것
  - 결국 둘다 라이브러리지만, 중간층은 결국 하위층을 wrap해서 쓰기 편하게 만든것
  - 누가 Driver API를 쓰나(하위층)?
    - 커널을 실행 중에 생성하는 경우: JIT 컴파일러, 딥러닝 프레임워크(트리톤 등)가 PTX를 만들어 바로 로드합니다.
    - 툴킷 없이 배포해야 하는 경우: nvcuda.dll은 드라이버에 이미 있으므로 cudart 없이도 동작합니다.
    - 컨텍스트를 정밀하게 제어해야 하는 경우

<br/>

  - 중간층 — Runtime API (파일 1개)
~~~c++
// runtime.cu
//   빌드: nvcc runtime.cu -o runtime.exe
#include <cuda_runtime.h>
#include <cstdio>

__global__ void WriteIndex(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}

int main() {
    const int n = 8;
    int h[n] = {};

    int* d = nullptr;
    cudaMalloc((void**)&d, sizeof(h));          // 할당

    WriteIndex<<<1, n>>>(d, n);                 // 런치

    cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    cudaFree(d);

    for (int i = 0; i < n; ++i) {
        std::printf("%d ", h[i]);
    }
}
~~~

<br/>

  - 하위층 — Driver API (파일 2개)
    - 커널을 따로 컴파일해서 실행 중에 불러와야 하므로 파일이 나뉜다

~~~c++
// kernel.cu  →  커널만 들어 있는 파일
//   빌드: nvcc -arch=sm_89 -ptx kernel.cu -o kernel.ptx
extern "C"                                      // 이름 맹글링 방지 (이름으로 찾아야 하므로)
__global__ void WriteIndex(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}
~~~

<br/>

~~~c++
// driver.cpp  →  .cu 가 아니어도 된다. nvcc 없이 MSVC 로만 빌드 가능
//   빌드: cl driver.cpp /I"%CUDA_PATH%\include" /link "%CUDA_PATH%\lib\x64\cuda.lib"
#include <cuda.h>
#include <cstdio>

int main() {
    const int n = 8;
    int h[n] = {};

    cuInit(0);                                  // ① 초기화 (Runtime 엔 없는 단계)
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, dev);                  // ② 컨텍스트를 손으로 생성

    CUdeviceptr d;                              // int* 가 아니라 전용 타입
    cuMemAlloc(&d, sizeof(h));                  // 할당

    CUmodule mod;
    cuModuleLoad(&mod, "kernel.ptx");           // ③ 커널을 파일에서 로드
    CUfunction f;
    cuModuleGetFunction(&f, mod, "WriteIndex"); // ④ 이름(문자열)으로 찾기

    int   nArg   = n;
    void* args[] = { &d, &nArg };               // ⑤ 인자를 void* 배열로 (타입 검사 없음)
    cuLaunchKernel(f,
                   1, 1, 1,                     // grid  x, y, z
                   n, 1, 1,                     // block x, y, z
                   0,                           // shared memory
                   nullptr,                     // stream
                   args, nullptr);              // 런치

    cuMemcpyDtoH(h, d, sizeof(h));
    cuMemFree(d);
    cuCtxDestroy(ctx);                          // ⑥ 정리도 손으로

    for (int i = 0; i < n; ++i) {
        std::printf("%d ", h[i]);
    }
}
~~~

<br/>

  - CUDA문법
    - __global__ : GPU에서 실행, CPU가 호출.
    - __device__ : GPU에서 실행, GPU가 호출.
    - __shared__ : 블록 안에서 공유하는 빠른 메모리.
    - <<<grid, block>>> : 몇 명이 실행할지.
    - threadIdx / blockIdx / blockDim : 내가 몇 번째 스레드인가.
    - __syncthreads() : 블록 안 스레드 집합 대기.

<br/>

  - GPU구조
    - SM(Streaming Multiprocessor) : 동시에 일하는 작은 프로세서
      - CPU 코어는 한 번에 한 가지 일을 하고, SM 하나는 수백 개의 일을 동시에 한다
      - SM의 수는 GPU마다 다름

```text
┌─────────────────── GPU (RTX 4060 Laptop) ───────────────────┐
│                                                              │
│   ┌─ SM 0 ─┐  ┌─ SM 1 ─┐  ┌─ SM 2 ─┐   ...   ┌─ SM 23 ─┐     │
│   │ 코어   │  │ 코어    │  │ 코어   │         │ 코어    │     │
│   │ 128개  │  │ 128개   │  │ 128개  │         │ 128개   │    │
│   │        │  │        │  │        │         │         │    │
│   │ 전용   │  │ 전용    │  │ 전용   │         │ 전용    │    │
│   │ 메모리 │  │ 메모리  │  │ 메모리 │          │ 메모리  │    │
│   └────────┘  └────────┘  └────────┘         └─────────┘    │
│                                                              │
│   ┌────────────────── VRAM 8 GB ──────────────────┐          │
│   │  모든 SM이 같이 쓰는 큰 메모리                   │         │
│   └────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

<br/>

```text
WriteIndex<<<8, 128>>>(d, n);
//            │   └─ 한 팀에 몇 명?   → 블록 크기
//            └───── 팀이 몇 개?      → 그리드 크기

int i = blockIdx.x * blockDim.x + threadIdx.x;
//      몇 번 팀     팀 인원      팀 내 몇 번째
//      (3)      ×   (128)    +   (5)          = 389번 일꾼
```

<br/>

  - 그래서 정리해보면

```text
VRAM               : 8.00 GB total / 6.93 GB free
warp size          : 32 threads
SM count           : 24
kernel timeout(TDR): ON (커널 2초 제한)

[블록 하나의 한계]
max threads/block  : 1024
max grid size      : 2147483647 x 65535 x 65535
shared mem/block   : 49152 bytes (48.0 KB)
registers/block    : 65536

[SM 하나의 수용량]
max threads/SM     : 1536  (warp 48개분)
max blocks/SM      : 24
registers/SM       : 65536  (스레드당 약 42개)
shared mem/SM      : 102400 bytes (100.0 KB)
→ GPU 전체 동시 상주 : 36864 threads (24 SM x 1536)
```

<br/>

  - 위의 데이터를 기반으로 설명한다
  - <<<블록수, 블록안에 스레드>>>
    - 블록안의 스레드 제한이 1024개
    - sm 1개당 넣을수 있는 블록이 24개, 1536스레드
    - warp size는 32개

<br/>

```text
블록 자리가 먼저 바닥

kernel<<<1000, 32>>>     // 블록 크기 32
블록 자리로  : 24개까지                  ← 여기서 막힘
스레드 자리로 : 1536 ÷ 32 = 48개까지

결과 : 블록 24개, 스레드 24 × 32 = 768명

블록 자리   : 24 / 24   ■■■■■■■■■■■■■■■■■■■■■■■■  꽉 참
스레드 자리 : 768 / 1536 ■■■■■■■■■■■■□□□□□□□□□□□□  절반만 참


스레드 자리가 먼저 바닥

블록 자리로  : 24개까지
스레드 자리로 : 1536 ÷ 256 = 6개까지      ← 여기서 막힘

결과 : 블록 6개, 스레드 6 × 256 = 1536명

블록 자리   : 6 / 24     ■■■■■■□□□□□□□□□□□□□□□□□□  여유 있음
스레드 자리 : 1536 / 1536 ■■■■■■■■■■■■■■■■■■■■■■■■  꽉 참

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

## 계산 예시 — block = 33

### 1단계 — warp 수로 바꾼다
33 = 32 + 1  →  warp 2개
                (두 번째 warp에 1명만. 31칸 낭비)


### 2단계 — 두 자리로 각각 계산해서 작은 쪽을 택한다
블록 자리로 : 24개까지
warp 자리로 : 48 / 2 = 24개까지
→ min(24, 24) = 24개

SM 하나에 블록이 24개 들어간다. 2개가 아니다.

### 3단계 — 결과
블록 24개
실제 스레드 : 24 x 33 = 792명
예약된 자리 : 24 x 2warp x 32 = 1536칸 (꽉 참)
정원 : 792 / 1536 = 51.6%


### 계산 순서 정리
① warp 수   = ceil(블록크기 / 32)
② 블록 수   = min( 24 , 48 / warp수 )     ← 나눗셈 결과는 내림
③ 스레드 수 = 블록수 x 블록크기
④ 정원 %    = 스레드수 / 1536


코어 128개는 어느 단계에도 나오지 않는다.
코어는 "몇 개가 들어가나"가 아니라 "한 클럭에 몇 명이 계산하나"를 정하는 값이다.
자리처럼 점유되는 것이 아니라 매 클럭 다시 비워진다.
```


