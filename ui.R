ui <- dashboardPage(
  skin="blue",
  
  #(1) Header
  
  dashboardHeader(title="Object Recognition App",#,style="font-size: 120%; font-weight: bold; color: white"),
                  titleWidth = 250,
                  tags$li(class = "dropdown"),
                  dropdownMenu(
                    type = "notifications", 
                    icon = icon("question-circle"),
                    badgeStatus = NULL,
                    headerText = "Feedback",
                    notificationItem("Send email to developer", icon = icon("file"),
                                     href = "mailto:liu.peng@u.nus.edu")
                  )),
  
  
  #(2) Sidebar
  
  dashboardSidebar(
    width=250,
    fileInput("input_image_upload","Upload image", accept = c('.jpg','.jpeg')),
    tags$br(),
    sliderInput("min_score_threshold","Confidence threshold",0,1,0.5),
    # tags$p("Upload the image here.")
    selectInput(inputId = "product_type",label = "Choose product",
                choices = c("Flour","Baby Food"),
                selected = NA),
    selectInput(inputId = "halal_status",label = "Halal status",
                choices = c("H","NH"),
                selected = NA),
    selectInput(inputId = "weight",label = "Choose weight",
                choices = c("50g","100g"),
                selected = NA),
    actionButton("submit","Submit",icon("paper-plane"), 
                 style="color: #fff; background-color: #337ab7; border-color: #2e6da4")
  ),
  
  
  #(3) Body
  
  dashboardBody(
    box(
      title = "Object Recognition", width = 12, solidHeader = TRUE, status = "primary",
      collapsible = T, collapsed = F,
      fluidRow(
        column(6,
               h4("Instruction:"),
               # tags$br(),
               tags$p("1. Upload image to be classified and set confidence threshold."),
               tags$p("2. Check prediction results."),
               tags$p("3. Select specific product category."),
               tags$p("4. Click submit to record in the system.")
        ),
        column(6,
               h4("Predicted Category:"),
               tableOutput("text")
        )
      ),
      
      fluidRow(
        column(h4("Image:"),imageOutput("output_image"), width=6),
        column(h4("Predicted Image:"),imageOutput("output_image2"), width=6)
      )
    ),
    box(
      title = "Image Gallery", width = 12, solidHeader = TRUE, status = "success",
      collapsible = T, collapsed = F,
      fluidRow(
        column(3,
               h3("All categories"),
               verbatimTextOutput("all_cats")
               ),
        column(3, 
               selectInput("input_image_select", "Select image",c("",ALL_IMGS),selected = ""),
               ),
        column(6,
               column(h4("Image:"),imageOutput("output_image_selected"), width=6),
               )
      )
    )
    
    # box(
    #   title = "Product Recording", width = 12, solidHeader = TRUE, status = "success",
    #   collapsible = T, collapsed = T,
    #   "test"
    # )
    
  ))