**Project 5: Independent HPC Research Sprint**

*Due: 4/29/2026 @ 12pm (noon)*
*Presentations: 4/28/2026 (In Class)*

**Topics**
* Scientific Computing Discovery
* GPU Algorithm Porting (CUDA/OpenACC/OpenMP Target)
* Performance Benchmarking & Profiling
* Technical Communication & LLM-Assisted Research

**Background**

While Project 4 introduced a fixed problem (N-Body) with a prescribed implementation, Project 5 flips the script. In the world of High Performance Computing (HPC), the first step is rarely writing code—it is identifying a problem in your domain of study (or personal interest) that is computationally constrained.

**Requirement Note:** This project is **Required for Graduate Students**. It is **Optional for Undergraduate Students** (who may submit it to replace their lowest final exam free response question grade).

**Project Overview: From Serial Bottleneck to GPU Acceleration**

You will identify a scientific research topic that you find genuinely interesting—this could be anything from modeling protein folding, simulating forest fire spread, analyzing financial option pricing, or even generating fractal art. 

Your task is to:
1. **Find a Code:** Locate an existing **serial** implementation of an algorithm that solves a *simple version* of this problem. (We strongly recommend starting with a problem size small enough to run on a laptop CPU in under 5 minutes, but large enough to show GPU benefit).
2. **Port to GPU:** Using one of the strategies discussed in class (e.g., massive thread parallelism, shared memory tiling, or CUDA stream concurrency), implement a parallel version.
3. **Benchmark:** Measure and graph the wall-clock time of the Serial CPU version vs. your Parallel GPU version.
4. **Document with AI:** Use a Large Language Model (LLM) to assist in drafting a 5-10 page technical report.
5. **Present:** Defend your approach and results in a 10-15 minute presentation on the last day of class.

**Functional Requirements**

* **Code Acquisition & Selection:**
    * You must start with a **working serial code** written in C/C++ (or Python with a clear C/C++ computational kernel you will rewrite).
    * **Constraint:** The problem should be relatively simple (e.g., 2D Heat Equation, Conway's Game of Life, Monte Carlo Pi Estimation with heavy iterations). *Overly complex solutions (e.g., full CFD solvers) are discouraged due to the time constraints of the semester.*

* **GPU Implementation:**
    * The parallel version must use **CUDA** (preferred) or a comparable GPU offloading strategy covered in class.
    * **Performance Strategy:** You must implement *at least one* performance optimization beyond the naive parallelization (e.g., using Shared Memory, reducing Atomic Operations, or overlapping Data Transfer with Compute).

* **Makefile & Repository Structure:**
    * Repository must contain **both** `serial` and `parallel` directories.
    * Each directory must contain a working **Makefile**.
    * Code must be pushed to the provided GitHub Classroom link.

* **LLM-Assisted Report (5-10 Pages):**
    * **Content:** Outline of the scientific problem, description of the serial algorithm, description of the GPU parallelization strategy, and analysis of the performance graph.
    * **LLM Citation:** You are required to use an LLM (ChatGPT, Claude, Gemini, DeepSeek, etc.) to assist with writing this report. **You must include an Appendix titled "LLM Usage Disclosure."** This appendix must include:
        * The name/model of the LLM used.
        * A list of the specific prompts you used.
        * A brief summary of how the output was edited/verified by you.

**Performance Benchmarking & Visualization**

You must produce a graph (using Python/Matplotlib, Excel, or Google Sheets) plotting:
* **X-Axis:** Problem Size (e.g., Grid Dimension, Number of Bodies, Number of Steps).
* **Y-Axis:** Execution Time (Seconds or Milliseconds).
* **Lines:** Serial CPU Time vs. Parallel GPU Time.

An image of this graph (`performance.png` or `performance.jpg`) must be included in the root of your GitHub repository.

**Presentation Requirements**

On the last day(s) of class, you will deliver a **10-15 minute** presentation covering:
1.  The "Why": What scientific question interests you here?
2.  The "How": How did you map this problem to GPU threads/blocks? (A simple diagram is expected).
3.  The "Speedup": Show your graph and explain the scaling behavior.
4.  The "Struggle": What was the hardest part of getting the code to compile/run correctly?

**Project Submission**

* **GitHub:** Link submitted via Canvas. Repository must contain:
    * `/serial/` (with source and Makefile)
    * `/parallel/` (with source and Makefile)
    * `report.pdf`
    * `performance_graph.png`
* **Compilation:** Any program lacking a correct Makefile or failing to compile with `make` will receive a grade of 0 for the coding portion.

**Pledged Work Policy**
* **Code:** The serial base code may be sourced from public scientific computing repositories (e.g., "Dr. Dobb's Journal examples" or university course webpages) but **must be cited clearly in the header comments.** The GPU porting logic must be your own work for this course.
* **LLM Use:** Use of LLMs for *code generation* is permitted **only** for boilerplate or debugging syntax errors; core parallel algorithm design must be your own. Use of LLMs for *report writing* is required (see above).

* Note: This project was written by DeepSeek (with modifications): https://chat.deepseek.com/share/7cmend3zo3jd3ugi3j
