For undergraduate students, understanding how to systematically conduct a research project is a missing piece in the curriculum. When I was an undergrad, I thought research was just doing a big course project, where the only thing I had to do was to finish the task assigned by my mentor, sort out engineering details, and deliver slightly better metrics than the previous work from some heuristic sparkles. It was not until I visited Stanford as an undergraduate research intern that I realized, instead of doing random trial and error, research could be manageable by standard procedures and methodologies.

This blog post is an overview of one of the most pivotal courses I took during my undergrad, [Stanford CS 197](https://web.stanford.edu/class/cs197/) instructed by [Prof. Bernstein](https://hci.stanford.edu/msb/). Here, I will share the key takeaways from the course, mainly how to identify a research problem and how to approach it properly. I hope this post helps you navigate a research project more efficiently and be less frustrated when you are stuck in the middle of the project.

## 1. What is Research?

**Research is finding a new way to solve a problem.**

The rise of deep learning is a great example. Before deep learning, people used hand-crafted features to design models (like SVM, Random Forest) to solve problems. However, the performance of these models was limited by the quality of the features as problems grew more complex. Then LeCun, Hinton, and Bengio proposed: "Let's drop all the features and let the model learn the features by itself by developing a general optimization algorithm." This new approach worked, thanks to the rise of computational power and the availability of big data. That's research: *proposing new perspectives* that flip old ideas to solve problems.

**Difference between research and engineering:**

Most of the time, an engineering problem is well-defined, and you need to find the right tools to tackle it. But for research, it must be innovative, implying that the exploration is highly uncertain and the outcome unpredictable. It's similar to the difference between visiting a tourist spot and exploring new land. The latter requires you to do something never done before, using your compass (your research advisor), your map (the literature review), and your brain (your creativity, hard work, and luck). Overall, research is a journey to reduce the uncertainty of the unknown.

## 2. How to Identify a Research Problem?

The first and most important step of a research project is to identify a research problem, ensuring you do not get lost in the unknown and work on something meaningful.

**Literature Review: Building the Map for Unknown Land**

To begin exploration, you need to build a map, which is the literature review. Start from one paper and use breadth-first search to find all related papers. The idea is simple: to see what has been done, what is missing, and why. After that, you might identify a few axes that can categorize the existing work and you can find a gap in the literature by identifying a missing piece in these axes. For example, you can refer to my [diffusion for control](https://github.com/jc-bao/diffuser-control-tutorial) as an example, where I divide existing work into three axes and find a gap in score computation, which leads to my research project [mbd](https://lecar-lab.github.io/mbd/). After the literature review, you should have a graph of papers or tables. This is your map to the unknown land and will form your paper's related work section.

**Define the Question: Deciding Where to Go**

After building the literature graph, you might have found some missing pieces of the puzzle. Now, you need to decide where to go, i.e., the *important* problem you try to solve and the possible *novel* solution you can propose. Specifically, the problem is the limitation you have identified from the literature (for feature-based models, the limitation is the quality of the features and the lack of generalization ability), and the solution is the outcome of your *new perspective* (for deep learning, the solution is to let the model learn the features by itself with backpropagation + deep neural network since we only care about the correctness of the output, not the correctness of the features).

This could be the hardest part of the research. Actually, the introduction of each paper is doing this: laying out the map in front of the reader, pointing out the missing piece that could have treasure (your important problem and its motivation), the major obstacles that prevent the solution (the bit), the bridge you built (how you flip the bit/method), and the treasure you found (the result).

## 3. How to Approach a Research Problem?

Now that we have built the map and decided on the destination, we need to figure out a way to get there. However, even with a rough idea of where to go, the landscape and possible obstacles are still unknown. This is where we need to develop two skills:

- **Vectoring**: Identify the most uncertain part of the project.
- **Velocity**: Develop a quick way to test the hypothesis so that we can move forward.

I personally really like these concepts by Prof. Bernstein. When moving through unknown territory, you should not devote too much time to planning a perfect route but instead engage in a fast trial-fail-adjust-trial cycle to find the right path. In other words, iteration >> planning.

**Vectoring: Deciding the Exact Next Move**

To decide the next move, you should ask one simple question: to make it work, what is the most essential part that is still missing? To do this, break down the components of your objective and identify which part could be the most important contribution of your work. For example, based on your hypothesis "deep neural networks can learn the features by themselves," the most important part is to study the capacity of the model and its relationship with the optimization algorithm/architecture rather than the normalization techniques.

In short, vectoring is about identifying the core problem (*high-impact and uncertainty*) and coming up with several assumptions (*precise and verifiable*).

**Velocity: Developing a Quick Way to Test the Hypothesis**

Since the hypothesis could be highly uncertain, it is always a bad idea to make a thorough implementation plan and aim for perfection. Instead, the objective at this stage is to gather as much information as possible in the shortest time rather than delivering a beautiful demo or impressive results.

To move fast, my suggestion is to fake everything you can and only keep the core part related to your hypothesis. This is especially true for modern deep learning projects, where the code workload is large and training a model could take days. Such tasks can distract you from the core issue and cause you to get stuck in the periphery.

Finding a good simplified abstraction problem, patching existing code, or narrowing down the hypothesis could be effective strategies. After testing the hypothesis, you can decide on the next vectoring based on the new information.

## 4. Conclusion

Research is quite similar to navigating in a new land; it is not a linear move towards the goal but an iterative process to explore the unknown and expand your understanding of the landscape. This is especially true for high-impact work, which moves further and reaches better goals by exploring more, failing more, and thus delivering more.

Hopefully, this post helps you be more productive, focused, and directed in your future research projects. Good luck and happy exploring!
