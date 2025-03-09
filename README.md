# M2 Coursework: LoRA Fine-Tuning for Time Series Forecasting

Please open the file `main.pdf` for the coursework instructions. Good luck and have fun!

## Clarifications

**Q: Are the differential equations used to generate the Lotka-Volterra data the standard equations or modified versions? Are the original parameters available and should they be used as model inputs?**

A:

The specific differential equations used to generate the data are not relevant. You should treat the dataset as observations you have been given and are now trying to model with a time series forecast approach.

The idea is that the LLM will pick up on the patterns from its input and be able to infer any relevant parameters in the underlying dynamics, similar to the approach in the LLMTIME paper.

This is similar to how LLMs work in practice. They predict the next token based on the previous sequence of tokens. They don't need structured inputs describing if the user is happy or sad. They just infer it all from the context.
