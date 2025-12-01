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
      plotOutput("outputPlot") 
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
    
    # Read the CSV file
    # data <- read.csv("output/output.csv", header = FALSE)
    # colnames(data) <- c("Position", "Skew")
    
    # # Get a color from the wesanderson palette
    # my_color <- wes_palette("GrandBudapest2", type = "discrete")[1]
    
    # # Create the plot in R using ggplot2
    # plot <- ggplot(data, aes(x = Position, y = Skew)) +
    #   geom_line(color = my_color) +
    #   labs(title = "Diagram", x = "Genome Position", y = "Skew Value")

    # # Save the plot as a PNG image
    # ggsave("output/diagram", plot = plot, width = 10, height = 6)
    
    # Render the plot in the Shiny app
    # output$outputPlot <- renderPlot({
    #   plot
    # })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
