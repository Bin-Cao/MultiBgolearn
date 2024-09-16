source code : [huggingface](https://huggingface.co/caobin/MultiBgolearn/tree/main)


# Structure and Flowcharts

This section provides a detailed overview of the MultiBgolearn algorithm's structure and workflow, using various diagrams to illustrate the different components and their relationships.

## System Architecture Diagram

The system architecture diagram shows the key modules of the MultiBgolearn algorithm and their connections.

```mermaid
graph TB
    subgraph MultiBgolearn[MultiBgolearn]
    DataPreprocessing[Data Preprocessing]
    ModelBuilding[Model Building]
    Prediction[Prediction]
    Optimization[Optimization]
    End[End and Output Results]
    end
    
    DataPreprocessing -->|Standardize Data| ModelBuilding
    ModelBuilding -->|Select Best Model| Prediction
    Prediction -->|Predict Virtual Space| Optimization
    Optimization --> End
```

## Data Preprocessing Flowchart

The data preprocessing flowchart outlines the steps involved in the data preprocessing module.

```mermaid
flowchart TB
    A[Start] --> B[Load Dataset]
    B --> C{Is the file CSV or Excel?}
    C -- Yes --> D[Split Data]
    C -- No --> E[Raise Error]
    D --> F[Standardize Features and Target]
    F --> G[Return Standardized Data]
    G --> H[End]
```

## Model Building Flowchart

The model building flowchart illustrates how to construct and select the optimal surrogate model.

```mermaid
flowchart TB
    A[Start] --> B[Load Training Data]
    B --> C[Select Model List]
    C --> D{For Each Model}
    D -- Leave-One-Out Cross-Validation --> E[Evaluate Model Performance]
    E --> F[Record R2 Score]
    F --> G[Select Model with Highest R2 Score]
    G --> H[End and Output Best Model]
```

## Prediction Flowchart

The prediction flowchart details the process of predicting the virtual space data using the selected model.

```mermaid
flowchart TB
    A[Start] --> B[Load Virtual Space Data]
    B --> C[Select Prediction Model]
    C --> D{Is the Model Gaussian Process?}
    D -- Yes --> E[Directly Predict Mean and Variance]
    D -- No --> F[Use Bootstrap to Predict Mean and Variance]
    E --> G[Return Prediction Results]
    F --> G
    G --> H[End]
```

## Optimization Flowchart

The optimization flowchart demonstrates how to apply the multi-objective Bayesian global optimization algorithm to recommend the optimal data points.

```mermaid
flowchart TB
    A[Start] --> B[Load Prediction Results]
    B --> C[Select Optimization Algorithm]
    C --> D{Execute Optimization}
    D --> E[Recommend Optimal Data Points]
    E --> F[Calculate Improvement Value]
    F --> G[Return Optimal Data Points and Improvement Value]
    G --> H[End]
```

## Logic Flowchart

The logic flowchart illustrates the overall logic flow of the MultiBgolearn algorithm.

```mermaid
flowchart TB
    A[Start] --> B[Data Preprocessing]
    B --> C[Model Building]
    C --> D[Prediction]
    D --> E[Optimization]
    E --> F[Output Recommended Data Points]
    F --> G[Output Improvement Value]
    G --> H[End]
```

## Monte Carlo Simulation Flowchart

The Monte Carlo simulation flowchart describes how to perform sampling using the Monte Carlo method.

```mermaid
flowchart TB
    A[Start] --> B[Define Mean and Variance]
    B --> C[Set Number of Samples]
    C --> D{For Each Sample}
    D -- Sampling --> E[Draw Samples from Multivariate Gaussian Distribution]
    E --> F[Collect All Samples]
    F --> G[End and Output Sample Set]
```

## Performance Evaluation Flowchart

The performance evaluation flowchart shows how to assess the predictive performance of the model.

```mermaid
flowchart TB
    A[Start] --> B[True vs Predicted Values]
    B --> C[Calculate Correlation Coefficient]
    C --> D[Compute Root Mean Squared Error]
    D --> E[Plot Scatter Diagram]
    E --> F[Save Plot]
    F --> G[End]
```