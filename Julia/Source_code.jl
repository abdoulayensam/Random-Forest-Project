using DataFrames
using Statistics
using Plots
using StatsPlots
using MLJ
using CSV
using StatsBase
using PrettyTables
using Random  

# --- Classes for Decision Tree and Random Forest ---
mutable struct DecisionTree
    max_depth::Int
    tree::Union{Dict{String,Any}, Any}

    DecisionTree(max_depth::Int=10) = new(max_depth, nothing)
end

mutable struct RandomForest
    n_trees::Int
    max_depth::Int
    sample_size::Int
    trees::Vector{DecisionTree}

    RandomForest(n_trees::Int=10, max_depth::Int=10, sample_size::Int=100) = new(n_trees, max_depth, sample_size, DecisionTree[])
end

#= ------------------ Utils Functions --------- =#
function entropy(y)
    counts = countmap(y)
    probabilities = [count / length(y) for count in values(counts)]
    return -sum(p * log2(p) for p in probabilities)
end

function split_data(X::DataFrame, Y::DataFrame; test_size=0.1, rng=Random.GLOBAL_RNG)
    random_state = check_random_state(rng)

    n_samples = nrow(X)
    n_train = n_samples - round(Int, test_size * n_samples)

    inds = randperm(random_state, n_samples)
    X = X[inds, :]
    Y = Y[inds, :]

    X_train = X[1:n_train, :]
    Y_train = Y[1:n_train, :]
    X_test  = X[n_train+1:end, :]
    Y_test  = Y[n_train+1:end, :]

    return  X_train, Y_train, X_test, Y_test
end

function information_gain(X, y, feature, threshold)
    feature_values = X[!, feature]
    left_indices = feature_values .<= threshold
    right_indices = .!left_indices
    
    left_y = y[left_indices]
    right_y = y[right_indices]
    
    if isempty(left_y) || isempty(right_y)
        return 0.0
    end

    total_entropy = entropy(y)
    left_entropy = entropy(left_y)
    right_entropy = entropy(right_y)

    left_weight = length(left_y) / length(y)
    right_weight = length(right_y) / length(y)
    
    return total_entropy - (left_weight * left_entropy + right_weight * right_entropy)
end

function find_best_split(X, y)
    best_gain = -1.0
    best_feature = nothing
    best_threshold = 0.0

    for feature in names(X)
        if eltype(X[!, feature]) <: Number  
            thresholds = unique(X[!, feature])
            for threshold in thresholds
                gain = information_gain(X, y, feature, threshold)
                if gain > best_gain
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                end
            end
        end
    end
    
    return best_feature, best_threshold
end

# --- Tree Building and Prediction Functions ---
function build_tree(X, y; depth=0, max_depth=10)
    if depth >= max_depth || all(y .== y[1]) || isempty(y)
        return StatsBase.mode(y)
    end

    best_feature, best_threshold = find_best_split(X, y)
    
    if isnothing(best_feature)
        return StatsBase.mode(y)
    end

    left_indices = X[!, best_feature] .<= best_threshold
    right_indices = .!left_indices

    left_tree = build_tree(X[left_indices, :], y[left_indices], depth=depth+1, max_depth=max_depth)
    right_tree = build_tree(X[right_indices, :], y[right_indices], depth=depth+1, max_depth=max_depth)

    return Dict{String,Any}(
        "feature" => best_feature,
        "threshold" => best_threshold,
        "left" => left_tree,
        "right" => right_tree
    )
end

function predict_row(row, tree)
    if !isa(tree, Dict)
        return tree
    end
    if row[tree["feature"]] <= tree["threshold"]
        return predict_row(row, tree["left"])
    else
        return predict_row(row, tree["right"])
    end
end

# --- Main Model Functions ---
function fit!(dt::DecisionTree, X, y)
    dt.tree = build_tree(X, y, depth=0, max_depth=dt.max_depth)
end

function predict(dt::DecisionTree, X)
    return [predict_row(row, dt.tree) for row in eachrow(X)]
end

function fit!(rf::RandomForest, X, y)
    empty!(rf.trees)
    n_samples = nrow(X)
    for _ in 1:rf.n_trees
        sample_indices = rand(1:n_samples, rf.sample_size)
        X_sample = X[sample_indices, :]
        y_sample = y[sample_indices]
        tree = DecisionTree(rf.max_depth)
        fit!(tree, X_sample, y_sample)
        push!(rf.trees, tree)
    end
end

function predict(rf::RandomForest, X)
    predictions = [predict(tree, X) for tree in rf.trees]
    return [StatsBase.mode([pred[i] for pred in predictions]) for i in 1:nrow(X)]
end

# --- Data Generation Function ---
function generate_dataset(n::Int)
    Random.seed!(42)  # Ensure reproducibility
    
    data = DataFrame(
        ID = 1:n,
        Age = rand(10:18, n),
        Sexe = rand(["M", "F"], n),
        Pref_Visuel = rand(0:10, n),
        Pref_Auditif = rand(0:10, n),
        Pref_Kinesthesique = rand(0:10, n),
        Math_Score = rand(40:100, n),
        Sciences_Score = rand(40:100, n),
        Langues_Score = rand(40:100, n),
        Temps_Etude_Visuel = round.(rand(1.0:0.1:10.0, n), digits=1),
        Temps_Etude_Auditif = round.(rand(1.0:0.1:10.0, n), digits=1),
        Temps_Etude_Kinesthesique = round.(rand(1.0:0.1:10.0, n), digits=1)
    )
    
    # Add learner type
    data.Apprenant_Type = map(eachrow(data)) do row
        preferences = Dict(
            "Visuel" => row.Pref_Visuel,
            "Auditif" => row.Pref_Auditif,
            "Kinesthesique" => row.Pref_Kinesthesique
        )
        max_pref = maximum(values(preferences))
        max_keys = [k for (k, v) in preferences if v == max_pref]
        
        return length(max_keys) > 1 ? "Mixte" : max_keys[1]
    end
    
    return data
end

# --- Plotting Functions ---
function plot_feature_importance(forest::RandomForest, feature_names)
    importance = zeros(length(feature_names))
    for tree in forest.trees
        if isa(tree.tree, Dict)
            accumulate_importance!(tree.tree, importance, feature_names)
        end
    end
    importance ./= sum(importance)
    
    # Create the plot with rotated x-axis labels
    p = bar(feature_names, importance,
           title="Feature Importance",
           xlabel="Features",
           ylabel="Importance",
           legend=false,
           xrotation=45,  
           size=(800, 400))  
    
    return p
end

function accumulate_importance!(node, importance, feature_names)
    if isa(node, Dict)
        feature_index = findfirst(==(node["feature"]), feature_names)
        importance[feature_index] += 1
        accumulate_importance!(node["left"], importance, feature_names)
        accumulate_importance!(node["right"], importance, feature_names)
    end
end

function plot_confusion_matrix(y_true, y_pred, class_names)
    cm = zeros(Int, length(class_names), length(class_names))
    for (true_idx, pred_idx) in zip(y_true, y_pred)
        cm[findfirst(==(true_idx), class_names), findfirst(==(pred_idx), class_names)] += 1
    end
    
    # Create annotations
    annotations = []
    for i in 1:length(class_names)
        for j in 1:length(class_names)
            push!(annotations, (j, i, string(cm[i, j])))
        end
    end
    
    # Create heatmap
    p = heatmap(cm,
                xticks=(1:length(class_names), class_names),
                yticks=(1:length(class_names), class_names),
                title="Confusion Matrix",
                xlabel="Predicted",
                ylabel="Actual",
                c=:blues,
                aspect_ratio=:equal,
                annotations=annotations)
    return p
end

# --- Classification Report Heatmap ---
function plot_classification_report(y_true, y_pred, class_names)
    # Calculate precision, recall, and F1-score for each class
    report = Dict()
    for (i, class_name) in enumerate(class_names)
        true_positives = sum((y_true .== class_name) .& (y_pred .== class_name))
        false_positives = sum((y_true .!= class_name) .& (y_pred .== class_name))
        false_negatives = sum((y_true .== class_name) .& (y_pred .!= class_name))
        
        precision = true_positives / (true_positives + false_positives + eps())
        recall = true_positives / (true_positives + false_negatives + eps())
        f1_score = 2 * (precision * recall) / (precision + recall + eps())
        
        report[class_name] = (precision=precision, recall=recall, f1_score=f1_score)
    end
    
    # Convert the report to a matrix for plotting
    metrics = ["Precision", "Recall", "F1-Score"]
    data = zeros(length(class_names), length(metrics))
    
    for (i, class_name) in enumerate(class_names)
        data[i, 1] = report[class_name].precision
        data[i, 2] = report[class_name].recall
        data[i, 3] = report[class_name].f1_score
    end
    
    # Create annotations
    annotations = []
    for i in 1:length(class_names)
        for j in 1:length(metrics)
            push!(annotations, (j, i, string(round(data[i, j], digits=2))))
        end
    end
    
    # Create heatmap
    p = heatmap(data,
                xticks=(1:length(metrics), metrics),
                yticks=(1:length(class_names), class_names),
                title="Classification Report",
                xlabel="Metrics",
                ylabel="Classes",
                c=:blues,
                aspect_ratio=:equal,
                annotations=annotations)
    
    return p
end

# --- Matrice de corrélation raffinée ---
function plot_correlation_matrix(df)
    numerical_features = select(df, Not([:ID, :Sexe, :Apprenant_Type]))
    corr_matrix = cor(Matrix(numerical_features))
    feature_names = names(numerical_features)
    
    # Create annotations
    annotations = []
    for i in 1:length(feature_names)
        for j in 1:length(feature_names)
            # Format the correlation value to two decimal places
            corr_value = round(corr_matrix[i, j], digits=2)
            # Use a larger font size for annotations
            push!(annotations, (j, i, text("$corr_value", :center, 10)))
        end
    end
    
    # Create heatmap 
    p = heatmap(corr_matrix,
                xticks=(1:length(feature_names), feature_names),
                yticks=(1:length(feature_names), feature_names),
                title="Correlation Matrix",
                xlabel="Features",
                ylabel="Features",
                c=:viridis, 
                aspect_ratio=:equal,
                annotations=annotations,
                xrotation=45,  
                size=(800, 600),  # Adjust plot size
                colorbar_title="Correlation",  # Add a color bar title
                clim=(-1, 1))  # Set color limits to ensure consistent scaling
    
    # Adjust the font size of the axis labels
    plot!(xguidefontsize=12, yguidefontsize=12, titlefontsize=14)
    
    return p
end

# --- Learner Type Distribution ---
function plot_learner_type_distribution(df)
    p = bar(countmap(df.Apprenant_Type),
           title="Learner Type Distribution",
           xlabel="Learner Type",
           ylabel="Count",
           legend=false)
    return p
end
# --- Distribution Plot Function ---
function plot_feature_distribution(df)
    features = ["Age", "Pref_Visuel", "Pref_Auditif", "Pref_Kinesthesique",
                "Math_Score", "Sciences_Score", "Langues_Score",
                "Temps_Etude_Visuel", "Temps_Etude_Auditif", "Temps_Etude_Kinesthesique"]
    
    plt = plot(layout=(3, 4), size=(1200, 900))
    for (i, feature) in enumerate(features)
        histogram!(df[!, feature], title=feature, subplot=i, legend=false)
    end
    return plt
end

# --- Display Decision Trees ---
function display_decision_trees(forest::RandomForest)
    for (i, tree) in enumerate(forest.trees)
        println("\nDecision Tree #$i:")
        print_tree(tree.tree)
    end
end

function print_tree(node, indent="")
    if !isa(node, Dict)
        println(indent, "Predict: ", node)
        return
    end
    println(indent, "Feature: ", node["feature"], " <= ", node["threshold"])
    print_tree(node["left"], indent * "  ")
    println(indent, "Feature: ", node["feature"], " > ", node["threshold"])
    print_tree(node["right"], indent * "  ")
end

function cross_validate(model_type, X, y, params)
    n_trees, max_depth, sample_size = params
    model = model_type(n_trees, max_depth, sample_size)
    scores = []
    for _ in 1:5  # 5-fold cross-validation
        indices = shuffle(1:nrow(X))
        train_indices = indices[1:Int(floor(0.8 * end))]
        test_indices = indices[Int(floor(0.8 * end)) + 1:end]
        
        X_train = X[train_indices, :]
        X_test = X[test_indices, :]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        fit!(model, X_train, y_train)
        y_pred = predict(model, X_test)
        accuracy = mean(y_pred .== y_test)
        push!(scores, accuracy)
    end
    return mean(scores)
end

function grid_search(model_type, X, y, param_grid)
    best_score = -1.0
    best_params = nothing
    
    # Generate all combinations of parameters
    param_combinations = Iterators.product(param_grid["n_trees"], param_grid["max_depth"], param_grid["sample_size"])
    
    scores = Dict()
    for params in param_combinations
        score = cross_validate(model_type, X, y, params)
        scores[params] = score
        
        if score > best_score
            best_score = score
            best_params = params
        end
        
        println("Parameters: n_trees=$(params[1]), max_depth=$(params[2]), sample_size=$(params[3])")
        println("CV Score: $score")
    end
    
    return best_params, best_score, scores
end

# --- Main Execution ---
function main()
    # Generate dataset
    n = 250
    df = generate_dataset(n)
    
    # Save to CSV
    # CSV.write("apprenants_dataset.csv", df)
    # println("Dataset saved to 'apprenants_dataset.csv'")
    
    # Print descriptive statistics
    println("\nDescriptive Statistics:")
    println(describe(df))

    # Plot distributions
    println("Generating correlation matrix...")
    p4 = plot_correlation_matrix(df)
    display(p4)
    savefig(p4, "correlation_matrix.png")  # Save plot

    # Generate feature distribution plot
    println("Generating feature distribution plot...")
    p4 = plot_feature_distribution(df)
    # display(p4)
    # savefig(p4, "feature_distribution.png")  # Save plot

    println("Generating learner type distribution...")
    p5 = plot_learner_type_distribution(df)
    # display(p5)
    # savefig(p5, "learner_type_distribution.png")  # Save plot
    
    # Prepare data for modeling
    X = select(df, Not([:ID, :Apprenant_Type]))
    y = df.Apprenant_Type
    
    # Define parameter grid for grid search
    param_grid = Dict(
        "n_trees" => [10, 20, 30],
        "max_depth" => [5, 10, 15],
        "sample_size" => [50, 100, 150]
    )
    
    # Perform grid search
    println("Performing grid search...")
    best_params, best_score, cv_scores = grid_search(RandomForest, X, y, param_grid)
    println("\nBest parameters: ", best_params)
    println("Best cross-validation score: ", best_score)

    # Train model with best parameters
    forest = RandomForest(best_params...)
    fit!(forest, X, y)
    
    # Make predictions
    y_pred = predict(forest, X)
    
    # Generate plots
     
    println("Generating feature importance plot...")
    p1 = plot_feature_importance(forest, names(X))
    # display(p1)
    # savefig(p1, "feature_importance.png")  # Save plot

    println("Generating confusion matrix...")
    class_names = unique(y)
    p2 = plot_confusion_matrix(y, y_pred, class_names)
    # display(p2)
    # savefig(p2, "confusion_matrix.png")  # Save plot

    println("Generating classification report heatmap...")
    p3 = plot_classification_report(y, y_pred, class_names)
    # display(p3)
    # savefig(p3, "classification_report_heatmap.png")  # Save plot


    # Display decision trees
    println("Displaying decision trees...")
    display_decision_trees(forest)

    # Calculate accuracy
    accuracy = mean(y_pred .== y)
    println("\nModel Accuracy: ", round(accuracy * 100, digits=2), "%")

    
    println("All plots and decision trees have been generated and saved.")
end


# Run the program
main()