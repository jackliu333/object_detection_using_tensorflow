ui <- fluidPage(
  tags$head(
    tags$style(HTML("
            .image-container {
                display: inline-block;
                width: 20%; /* Adjust the width as needed */
                padding: 1%; /* Provides spacing between images */
            }
            .row {
                text-align: center;
            }
            .narrow-select {
                width: 150px; /* Adjust the width of the select boxes */
            }
        "))
  ),
  titlePanel("Image Viewer Based on Labels"),
  sidebarLayout(
    sidebarPanel(
      selectInput("productType", "Product Type", choices = all_prdtypes, selected = all_prdtypes[1]),
      selectInput("weight", "Weight", choices = all_weights, selected = all_weights[1]),
      selectInput("halal", "Halal", choices = all_halals, selected = all_halals[1]),
      selectInput("healthy", "Healthy", choices = all_healths, selected = all_healths[1]),
      tableOutput("filtered_tbl")      
      # actionButton("search", "Search")
    ),
    mainPanel(
      uiOutput("imageGallery")
      # uiOutput("testImage")
      
    )
  )
)
