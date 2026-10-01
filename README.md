###### Top

  - [프로젝트](#프로젝트)
  - [CUDA기초](#cuda기초)_S1_ExecModel -> 01_VectorAdd
  - [Debug vs Release,Pcle,대역폭](#debug-vs-releasepcle대역폭)
  - [Nsight](#nsight)
  - [SISD,SIMD,SIMT](#sisdsimdsimt)
  - [2D,3D커널](#2d3d커널)_S1_ExecModel -> 02_BgrToGray
  - [비동기,에러](#비동기에러)_S1_ExecModel -> 03_AsyncAndErrors
  - [메모리접근속도,공유메모리](#메모리접근속도공유메모리)_S2_Memory -> 04_Transpose


2D 커널

<br/>
<br/>

***

#프로젝트
  - [CudaPrectice](https://github.com/BuMinKyoo/CudaPrectice)


###### [프로젝트](#프로젝트)
###### [Top](#top)

<br/>
<br/>

***

# CUDA기초
  - S1_ExecModel -> 01_VectorAdd

<br/>

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

<br/>

  - 내장된 변수
    - gridDim.x : 블록이 몇 개인가
    - blockIdx.x : 내가 몇 번 블록인가
    - blockDim.x : 블록 하나에 스레드가 몇 개인가
    - threadIdx.x : 블록 안에서 내가 몇 번째인가

###### [CUDA기초](#CUDA기초)
###### [Top](#top)

<br/>
<br/>

***

# Debug vs Release,Pcle,대역폭

  - Debug와 Release를 했을때 순위가 완전히 뒤집힌다
  - Debug 빌드에서는 두 가지 최적화가 동시에 꺼진다.
    - 호스트 코드 (cl.exe)  : /Od  → CPU 루프가 최적화 안 됨      → CPU loop 15배 느림
    - 디바이스 코드 (nvcc)  : -G   → 커널 명령어가 최적화 안 됨   → 커널 최대 2.7배 느림

<br/>

```text
항목                  Debug        Release      차이
─────────────────────────────────────────────────────
CPU loop             52.0 ms       3.6 ms      15배
block 8               4.99 ms      2.26 ms      2.2배
block 16              2.50 ms      1.14 ms      2.2배
block 32              1.15 ms      0.61 ms      1.9배
block 128             0.54 ms      0.52 ms      거의 같음
block 256             0.57 ms      0.52 ms      거의 같음
block 1024            0.68 ms      0.50 ms      1.4배 (역전)
[B] grid 1           36.0 ms      12.5 ms       2.7배
H2D 복사              7.0 ms       6.6 ms      거의 같음
D2H 복사              3.3 ms       3.7 ms      거의 같음
```
<br/>

  - Debug, Release 문제 뿐만 아니라, 대역폭에 따라서도 다양한 변수가 발생한다

<br/>

```text
 블록크기 블록당warp  SM당블록  SM당warp  SM당스레드  점유율     디버그     릴리즈     대역폭  달성률
--------- ----------- --------- --------- ----------- ------- --------- --------- --------- -------
        8           1        24        24         192   12.5%   4.99 ms   2.26 ms   53 GB/s     21%
       16           1        24        24         384     25%   2.50 ms   1.14 ms  105 GB/s     41%
       32           1        24        24         768     50%   1.15 ms   0.61 ms  197 GB/s     77%
      128           4        12        48        1536    100%   0.54 ms   0.52 ms  231 GB/s     90%
      256           8         6        48        1536    100%   0.57 ms   0.52 ms  231 GB/s     90%
     1024          32         1        32        1024     67%   0.68 ms   0.50 ms  240 GB/s     94%
   grid 1          8*         1         8         256    0.7%  36.00 ms  12.50 ms  9.6 GB/s      4%

다음 표를 보면 256에서 sm의 점유율이 100%인데, 1024일때인 67%보다 느린것을 확인할 수 있다(릴리즈 에서)
이는 이미 67%의 점유율 정도만으로 vram의 대역폭이 가득 찼음을 의미한다.

-> 대역폭은 총 2가지가 있는대
[PCIe]  CPU RAM ↔ GPU VRAM       약  12 GB/s   ← cudaMemcpy 가 쓰는 길
[VRAM]  GPU 코어 ↔ GPU VRAM      약 256 GB/s   ← 커널이 a[i] 읽을 때 쓰는 길
여기서 말한 대역폭은 2번째를 의미한다, 즉 점유율만 100%로 하는게 좋은것이 아니다

-> 대역폭 계산하기
10,000,000의 N을 계산한다고 하면
GPU안에서는 a+b = c 계산을 하기 때문에 각각 float으로써 4바이트가 된다, 즉 3개를 건들기 때문에
1회 계산에 12바이트를 건들게 되고
이건, 10,000,000 x 12 = 120,000,000바이트를 건드는 것이 된다. 이걸 전부 계산한 것이, block 1024 기준으로 했을때 0.000498초~0.0005초 사이 임으로,
바이트 ÷ 시간 = 대역폭    120,000,000 바이트 ÷ 0.000498초 = 241,000,000,000 바이트/초 가 된다

-> 결론
- Release : 커널이 순수 메모리 병목 → 점유율 67%로도 대역폭이 포화되므로
            블록 수가 적은(9,766 vs 39,063) 1024 가 스케줄링 비용에서 이긴다
- Debug   : -G 로 명령어가 늘어 연산 비중이 커짐 → 대기를 가릴 warp 가 더 필요 →
            점유율 100%인 256 이 이긴다

```

<br/>

  - 위에서 계산한 초당 몇 바이트를 옮길수 있는지 메모리 대역폭을 아래에서 확인할 수 있다
    - 1번 사진 : VRAM 대역폭(GPU 코어 ↔ GPU 메모리)
    - 2번 사진 : PCIe 대역폭(CPU ↔ GPU)

<br/>

<img width="1379" height="773" alt="image" src="https://github.com/user-attachments/assets/746290a2-1b0f-4534-89d4-4491010c3df1" />

<br/>

  - Gen4 x8
<img width="918" height="774" alt="image" src="https://github.com/user-attachments/assets/f75da92b-03fe-4a95-a425-64820fcef9e6" />

<br/>

```text
Gen4 : 레인 하나당 약 2 GB/s
x8   : 레인 8개
────────────────────────
이론 최대 : 약 16 GB/s
```

<br/>

  - 현재까지의 병목관련
    - PCIe 전송 대역폭 — 가장 비싸다 (커널의 20배)
    - VRAM 대역폭(GPU 칩 ↔ VRAM 메모리 칩)
    - warp를 못 채우는 것
    - grid가 작아서 SM을 다 못 쓰는 것(하나의 블록은 하나의 sm에 다 들어가야함)
    - sm안에 블록수제한과 쓰레드수 제한을 최대한으로 못채우는것(블록하나의 쓰레드도 조절해야함)
      - 이걸 100%채우는걸로 목표할 필욘 없음, 대역폭이 가득찬 이후에는 여러요소로 달라질수 있음

###### [Debug vs Release,Pcle,대역폭](#Debug-vs-ReleasePcle대역폭)
###### [Top](#top)


<br/>
<br/>

***

# Nsight
  - NVIDIA에서 공식 제공하는 GPU 프로파일링 및 디버깅 도구 모음

<br/>

  - 비쥬얼스튜디오 에서 확장 추가

<br/>

<img width="828" height="490" alt="image" src="https://github.com/user-attachments/assets/eba3e2f2-1218-4f45-8858-cd101ff5f4d9" />

<br/>

  - Nsight Systems(전체 흐름 & 병목 탐색용 - nsys) 설치
    - 역할: 거시적(Macro) 관점에서 CPU와 GPU 간의 상호작용을 타임라인으로 보여줌
    - CPU 연산, cudaMemcpy, 커널 실행이 시간에 따라 어떻게 맞물려 있는지
    - GPU가 작업을 기다리느라 멈춰 있는 구간(Idle Time)이 어디인지 등
  - Nsight Compute (단일 커널 정밀 분석용 - ncu) 설치
    - 역할: 미시적(Micro) 관점에서 특정 CUDA 커널 내부의 실행 효율을 정밀하게 분석
    - 하드웨어 점유율(SM Occupancy), 연산 처리량(Compute Throughput), 메모리 대역폭 사용량
    - 레지스터 스필(Register Spill)이나 공유 메모리(Shared Memory) 뱅크 충돌 여부
    - 해당 커널이 메모리 바운드(Memory-bound)인지, 연산 바운드(Compute-bound)인지 진단

<br/>

  - Nsight Visual Studio Edition (디버거) 설치

###### [Nsight](#nsight)
###### [Top](#top)


<br/>
<br/>

***

# SISD,SIMD,SIMT
  - SISD (Single Instruction, Single Data)
    - 명령어 하나가 데이터 하나를 처리
    - 전통적인 CPU 코어 하나의 기본 동작

<br/>

  - SIMD (Single Instruction, Multiple Data)
    - 명령어 하나가 데이터 여러 개를 한꺼번에 처리
    - CPU의 SSE, AVX, NEON이 여기에 해당
    - 예를들어 -> 레지스터 자체가 넓어서(AVX는 256비트) float 8개를 한 레지스터에 담고, 명령 하나로 8개를 동시에처리
    - if처럼 원소마다 다른 분기가 필요하면 마스크를 직접 만들어 처리해야 한다

<br/>

  - SIMT (Single Instruction, Multiple Threads)
    - GPU동작
    - 하드웨어는 SIMD처럼 동작하지만, 프로그래머는 스레드 하나의 코드만 쓴다
    - SIMD와 다른 점
      - 각 스레드가 자기 레지스터, 자기 인덱스, 자기 실행 흐름을 가진다, 코드는 스레드 하나 기준의 평범한 스칼라 코드이고, 32개씩 묶는 일은 하드웨어가 알아서 한다
      - 스레드마다 임의의 주소를 읽고 쓸 수 있다
      - 분기를 그냥 if로 쓸 수 있다. 하드웨어가 마스크를 자동으로 처리해 준다
        - SIMD 코드에서는 변수 하나에 값이 8개 들어 있어서, 조건의 답도 8개가 나온다는 뜻, if는 참/거짓 하나를 받아서 프로그램 전체를 한쪽으로 점프시키는 문법이라, 답 8개를 받을 수가 없다
        - 이렇게 양쪽 경로를 다 도는 비용을 "워프 발산" 이라고 한다

~~~c
if (threadIdx.x % 2 == 0)  A();
else                        B();

이렇게 분기를 타게 되면, 
짝수 스레드는 A, 홀수 스레드는 B로 갈라지면, 하드웨어는 A를 실행하는 동안 홀수 스레드를 꺼두고, 그다음 B를 실행하는 동안 짝수 스레드를 꺼둔다
두 경로를 순서대로 모두 실행하게 되어 이 구간은 효율이 절반으로 떨어집니다. 결국 속을 들여다보면 SIMD라서 생기는 현상
~~~

<br/>

  - 여기서 더 중요한 부분이 있음 if문이 있다고 해서 발산을 하냐? 그건 아님
  - 발산은 같은 warp 안의 스레드 32개가 if에서 서로 다른 길로 갈릴 때만 일어난다, wrap으로 묶이는 갯수는 엔비디어 칩마다 다름 보통 코어안의 파티션으로 묶이는 부분과 같음
  - warp안에서는 명령이 하나이고, 같은 회로를 거침 -> warp 안에서 if 결과가 섞이면 명령이 둘로 갈림

<br/>

~~~c
__global__ void kernel(float* a) {
    int tid = threadIdx.x;
    if (조건) A();   // A 경로
    else      B();   // B 경로
}

// 조건에 넣는것이,

/*
tid < 32 일 경우
warp 0 (0~31) : A A A A ... A A   → 전부 A
warp 1 (32~63): B B B B ... B B   → 전부 B
발산 없음
*/


/*
tid < 16 일 경우
warp 0 (0~31) : A A ... A (0~15) | B B ... B (16~31)   → 섞임
warp 1 (32~63): B B B B ... B B                         → 전부 B
warp 0은 발산합니다
*/
~~~

###### [SISD,SIMD,SIMT](#sisdsimdsimt)
###### [Top](#top)


<br/>
<br/>

***


# 2D,3D커널
  - S1_ExecModel -> 02_BgrToGray
  - 01_VectorAdd 프로젝트는 1D이고, 02_BgrToGray는 2D,3D까지 다루고 있음

<br/>

~~~c
dim3 block(16, 16);
dim3 grid(DivUp(w, 16), DivUp(h, 16));
BgrToGray<<<grid, block>>>(dSrc, dDst, w, h);

/*
dim3는 그냥 리스트 3개짜리인것
2d,3d는 위와 같은 코드로 집어 넣는것이고, 위의 코드는 3d이지만 마지막 z축이 1이라서 감춰짐
*/

// 커널에서 사용할때 아래와 같이 사용하면됨
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x >= w || y >= h) return;
~~~

###### [2D,3D커널](#2d3d커널)
###### [Top](#top)

<br/>
<br/>

***

# 비동기,에러
  - _S1_ExecModel -> 03_AsyncAndErrors

```text
HeavyWork<<<grid, block>>>(d, n, iters);

// 이와 같이 cpu가 gpu에게 작업을 전달하는건 작업을 마치 주문만 하고 돌아오는것
// cpu는 gpu에게 작업 주문만 한다
```

<br/>

  - cudaDeviceSynchronize
    - gpu작업이 다 끝날때까지 cpu가 멈춰서 있다

<br/>

```text
// CPU가 먼저 끝나는 경우
CPU  [런치][CPU 50ms][─ 30ms 기다림 ─]
GPU  [───────── GPU 80ms ─────────]
                                   ↑ afterSync = 80ms
// GPU가 먼저 끝나는 경우
CPU  [런치][──── CPU 120ms ────][Sync 즉시 통과]
GPU  [─── GPU 80ms ───]
                                ↑ afterSync = 120ms
```

<br/>

  - 엔비디어 커널런치는 조용히 실패함 -> 반환 값을 받을 자리가 없기 때문
    - 에러가 발생하면, 마지막에러 슬롯에 적어 놓는다, 우린 그걸 빼서 확인하면 됨

<br/>

  - cudaEventRecord : cpu가 gpu에게 너 이거 보면 시간 찍어 라고 큐에 집어 넣고 나온다, gpu는 자기 할일 하다가 이명령어를 큐에서 빼면 시간을찍음
  - cudaEventSynchronize(end_) : end 표식이 찍힐 때까지 대기

<br/>

~~~c
cudaError_t e1 = cudaPeekAtLastError();   // 보기만
cudaError_t e2 = cudaPeekAtLastError();   // 또 보기
cudaError_t e3 = cudaGetLastError();      // 꺼내기 (지워짐)
cudaError_t e4 = cudaGetLastError();      // 다시 꺼내기

Peek #1 : cudaErrorInvalidConfiguration
Peek #2 : cudaErrorInvalidConfiguration      ← 여전히 남아 있음
Get  #1 : cudaErrorInvalidConfiguration  (invalid configuration argument)
Get  #2 : cudaSuccess                        ← 지워짐!
~~~

<br/>

  - 여기서 중요한것은 커널이 조용히 실패한 후에, 다른 커널들은 진행이 잘 된긴 하지만, CUDA_CHECK_LAUNCH()를 하는 순간 이전에 있던 에러가 튀어 나오면서 프로그램이 exit됨
  - 따라서 cuda함수를 사용한 후에는 항상 CUDA_CHECK_LAUNCH를 사용해서 어떤 함수에서 실패했는지 정확히 체크해야함


###### [비동기,에러](#비동기에러)
###### [Top](#top)


<br/>
<br/>

***

# 메모리접근속도,공유메모리
  - S2_Memory -> 04_Transpose

<br/>

  - GPU메모리구조는 아래와 같다
    - gpu는 VRAM이라는 전역메모리가 있고, SM안에 전용메모리가 따로 있음
    - SM안에 전용메모리는 블록별로 할당받는 식으로 진행됨

```text
           VRAM 8 GB            GPU 칩 "바깥" 기판 위 GDDR6     ← cudaMalloc
-> VRAM에서 사용하는 메모리는 느림

공유 메모리 96KB/SM(칩마다 다름)  GPU 칩 "안쪽" 각 SM의 SRAM     ← __shared__
-> 각 sm안에서 가지고 있는 공유메모리, 이거 전체를 블록이 다같이 쓰진 않는다
-> 블록이 SM에 올라가는 순간 자기 몫을 떼어 받고, 블록이 끝나면 반납해서 다음 블록이 그 자리를 씁니다.
-> 1블록이 할당받을수 있는 한계는 또 따로 있음 __shared__ 를 하면 그많큼 할당됨(한계 이상은 안됨)

마지막으로 가장빠른 메모리는 레지스터가 있음
-> 레지스터는 스레드의 지역 변수가 사는곳임
__global__ void TransposeShared(const float* in, float* out, int n) {
    int x = ...;       // ← 레지스터
    int y = ...;       // ← 레지스터
    for (int j = ...)  // ← 레지스터
-> 스레드마다 자기것을 따로 가짐
-> 블록이 SM에 올라갈 때 모든 스레드의 레지스터가 물리적으로 할당되고 블록이 끝날 때까지 유지된다.
-> CPU처럼 저장/복원할 게 없어서 전환이 공짜인 이유가됨
-> 레지스터의 한계는 블록당 존재하고, sm당 존재함
-> 상황1 : SM하나에 블록제한, 쓰레드 제한 있는것처럼 레지스터 제한도 있음 -> 레지스터가 먼처 차면 활용이 떨어진다는것
-> 상황2 : 레지스터가 하드웨어 한도를 넘으로 "레지스터 스필"이 일어남 -> 이걸 VRAM에 저장하기 때문에 속도가 현져히 떨어짐

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

///////////////////////////////////////////

┌─ CPU 쪽 ─────────────────────────────────┐
│  시스템 RAM                                │
│    std::vector<float> hIn                 │
└───────────────────┬───────────────────────┘
                    │  ← ① cudaMemcpy(H2D)   PCIe,  약 12 GB/s
                    ↓
┌─ GPU 그래픽카드 ──────────────────────────────────────────┐
│                                                           │
│   ┌─ SM 0 ──────┐  ┌─ SM 1 ──────┐   ...                  │
│   │ 코어        │  │ 코어        │                        │
│   │ 레지스터    │  │ 레지스터    │                         │
│   │ __shared__  │  │ __shared__  │  ← ③ 커널만 채울 수 있음 │
│   └──────┬──────┘  └──────┬──────┘                        │
│          │                │                               │
│          └────────┬───────┘   ← ② 커널이 읽고 씀, 256 GB/s │
│                   ↓                                       │
│   ┌──────────── VRAM 8 GB ─────────────┐                  │
│   │   cudaMalloc 이 잡는 곳             │  ★ cudaMemcpy 의 │
│   │   dIn, dOut                        │     목적지        │
│   └────────────────────────────────────┘                  │
└───────────────────────────────────────────────────────────┘

///////////////////////////////////////////

┌──────────── SM 0 의 공유 메모리 (물리적으로 96KB 하나) ────────────┐
│                                                                    │
│  ┌─ 블록A ─┐ ┌─ 블록B ─┐ ┌─ 블록C ─┐ ┌─ 블록D ─┐      (빈 공간)    │
│  │  4.2KB  │ │  4.2KB  │ │  4.2KB  │ │  4.2KB  │                   │
│  │ A만 봄  │ │ B만 봄  │ │ C만 봄  │ │ D만 봄  │                   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                   │
│       ↑                                                            │
│   서로 넘볼 수 없다 (칸막이)                                         │
└────────────────────────────────────────────────────────────────────┘

```

<br/>

  - 메모리 심화버전
    - VRAM
      - VRAM은 바이트 단위로 꺼내 쓸 수 없다. 최소 32바이트 덩어리로만 주고받는다. -> 메모리는 항상 덩어리로 움직인다
      - 읽거나 쓸때 연속된 메모리가 아닌경우는, VRAM에서 데이터 읽어오기를 여러번 해야하기 때문에 더 느려짐
      - 내가 만약 쓰레드 100개를 돌리는데, list a[300] 이렇게 된 데이터를 3바이트씩 떨어진 데이터를 읽으려고한다면, 1회 vram에서 읽어오는데 32바이트이니까 1회읽어오면 10개 정도의 쓰레드가 값을사용할수 있고 총 10회 정도 읽어야 한다
      - 그리고 더 정확하게는 VRAM에서 읽을때는 WARP단위로 읽게 된다 100개의 쓰레드이니, WRAP이 0~3 으로 총 4개, 만약 0번째 WARP이면 0~31번째 쓰레드 이고 여기는 0~95번까지 데이터가 필요하므로 VRAM에서 총 3회를 읽어와야함, 이렇게 따져보면 0번째 WARP: 3회, 1번째 WARP: 3회, 2번째 WARP: 3회, 3번째 WARP: 1회 이렇게 총 10회가 된다
    - 공유메모리
      - BANK단위로 되어 있음
      - 32개의 독립된 조각으로 쪼개고, 한 클럭에 4바이트 하나씩 내줌
      - 따라서 VRAM처럼 1바이트 읽으려고 32바이트 읽어오는짓을 하지 않은, 메모리 낭비가 없다
      - 하지만, 같은 BANK에 몰리면 병목이 생김
      - BANK충돌에 관련되서는 나중에 다시 공부하기



<br/>

  - 현재까지의 병목관련
    - PCIe 전송 대역폭 — 가장 비싸다 (커널의 20배)
    - VRAM 대역폭(GPU 칩 ↔ VRAM 메모리 칩)
    - warp를 못 채우는 것
    - grid가 작아서 SM을 다 못 쓰는 것(하나의 블록은 하나의 sm에 다 들어가야함)
    - sm안에 블록수제한과 쓰레드수 제한을 최대한으로 못채우는것(블록하나의 쓰레드도 조절해야함)
      - 이걸 100%채우는걸로 목표할 필욘 없음, 대역폭이 가득찬 이후에는 여러요소로 달라질수 있음
    - SM안에 레지스터수 제한이 있음
    - VRAM, 공유메모리 속도 차이


###### [메모리접근속도,공유메모리](#메모리접근속도공유메모리)
###### [Top](#top)






