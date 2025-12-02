# Check if shiny is installed; if not, install it
if (!require("shiny")) {
  install.packages("shiny")
}

# Check if ggplot2 is installed; if not, install it
if (!require("ggplot2")) {
  install.packages("ggplot2")
}

if (!require("wesanderson")) {
  install.packages("wesanderson")
}

library(shiny)
library(ggplot2)
library(wesanderson)

ui <- fluidPage(
  titlePanel("Influenza Autoregression Analysis"),
    
  sidebarLayout(
    # Dropdown to select country
    sidebarPanel(
      selectInput(
        inputId = "country", # Unique ID for the input
        label = "Select a Country for the analysis:", # Label displayed to the user
        choices = c("India", "USA", "China") # List of available choices
      ),
      actionButton("runGoCode", "Run Autoregression")
    ),
    
    mainPanel(
      plotOutput("outputPlot"),
      plotOutput("grangerPlot"),
      plotOutput("irfPlot")
    )
  )
)
 
server <- function(input, output) {
  
  observeEvent(input$runGoCode, {
    # Determine the country selected by the user
    country <- input$country
    
    # Compile the Go program
    compile_result <- system("go build", intern = TRUE)
    print(compile_result)  # For debugging, to see compile output
    
    # Run the compiled Go program
    run_result <- system(paste("./application ", as.character(country)), intern = TRUE)
    print(run_result)  # For debugging, to see runtime output
    
    # Read the CSV files
    forcast_data <- read.csv("../Files/Output/forcast_results.csv")
    granger_data <- read.csv("../Files/Output/granger_results.csv")
    irf_data <- read.csv("../Files/Output/irf_results.csv")


    # Add a Week variable if not present
    forcast_data$Week <- seq_len(nrow(forcast_data))

    # Reshape data for plotting
    library(reshape2)
    data_long <- melt(forcast_data, id.vars = "Week", measure.vars = c("inf_a_log_diff", "inf_b_log_diff"),
                      variable.name = "Virus", value.name = "LogDiff")
    
    # TODO: last 30 day, next 10 - pediction
    # Plot only Influenza A and B log diffs
    plot <- ggplot(data_long, aes(x = Week, y = LogDiff, color = Virus)) +
      geom_line(size = 1) +
      geom_point(size = 2) +
      labs(title = paste("Influenza A and B Log Diff Over Time in", country),
           x = "Week",
           y = "Log Difference",
           color = "Virus") +
      theme_minimal()

    # TODO Make cooler
    # irf results into bar chart
    plot_irf <- ggplot(irf_data, aes(x = ShockVariable, y = CumulativeImpact, fill = ShockVariable)) +
      geom_bar(stat = "identity") +
      labs(title = paste("Impulse Response Function Results in", country),
           x = "Variable",
           y = "Impact") +
      theme_minimal()

    # TODO Make cooler
    # granger causality results
    plot_granger <- ggplot(granger_data, aes(x = CauseVar, y = FStatistic, fill = CauseVar)) +
      geom_bar(stat = "identity") +
      labs(title = paste("Granger Causality Test Results in", country),
           x = "Variable",
           y = "F-Statistic") +
      theme_minimal()
    

    # Render the plot in the Shiny app
    output$outputPlot <- renderPlot({
      plot
    })
    output$grangerPlot <- renderPlot({
      plot_granger
    })
    output$irfPlot <- renderPlot({
      plot_irf
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
