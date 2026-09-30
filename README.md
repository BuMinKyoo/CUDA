###### Top

  - [프로젝트](#프로젝트)
  - [CUDA기초](#cuda기초)
  - [Debug vs Release,Pcle,대역폭](#debug-vs-releasepcle대역폭)
  - [Nsight](#nsight)
  - [SISD,SIMD,SIMT](#sisdsimdsimt)
  - [2D,3D커널](#2d3d커널)


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






