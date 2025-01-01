# basic_auth.jl
#test
using Gtk

# Hardcoded credentials for testing
const VALID_USERNAME = "admin"
const VALID_PASSWORD = "password123"

function create_auth_dialog()
    # Create the main window
    win = GtkWindow("Authentication Required")
    set_gtk_property!(win, :modal, true)
    set_gtk_property!(win, :window_position, 3)  # :center
    set_gtk_property!(win, :default_width, 300)
    set_gtk_property!(win, :default_height, 200)
    
    # Create a vertical box for layout
    box = GtkBox(:v)
    set_gtk_property!(box, :spacing, 10)
    set_gtk_property!(box, :margin, 15)
    push!(win, box)
    
    # Add a warning message at the top
    warning_label = GtkLabel("Authentication is required.\nThis window cannot be closed until valid credentials are provided.")
    set_gtk_property!(warning_label, :justify, 2)  # Center justify
    set_gtk_property!(warning_label, :wrap, true)
    push!(box, warning_label)
    
    # Username field
    username_label = GtkLabel("Username:")
    set_gtk_property!(username_label, :halign, 0)  # align start
    username_entry = GtkEntry()
    set_gtk_property!(username_entry, :hexpand, true)
    
    # Password field
    password_label = GtkLabel("Password:")
    set_gtk_property!(password_label, :halign, 0)  # align start
    password_entry = GtkEntry()
    set_gtk_property!(password_entry, :visibility, false)  # hide password
    set_gtk_property!(password_entry, :hexpand, true)
    
    # Result label for showing authentication status
    result_label = GtkLabel("")
    set_gtk_property!(result_label, :hexpand, true)
    
    # Login button
    login_button = GtkButton("Login")
    
    # Add all components to the box
    push!(box, username_label)
    push!(box, username_entry)
    push!(box, password_label)
    push!(box, password_entry)
    push!(box, login_button)
    push!(box, result_label)
    
    # Authentication result variable
    auth_result = Ref(false)
    
    # Handle button click
    signal_connect(login_button, "clicked") do widget
        username = get_gtk_property(username_entry, :text, String)
        password = get_gtk_property(password_entry, :text, String)
        
        if username == VALID_USERNAME && password == VALID_PASSWORD
            set_gtk_property!(result_label, :label, "Authentication successful!")
            auth_result[] = true
            sleep(1)  # Give user time to see the success message
            destroy(win)
        else
            set_gtk_property!(result_label, :label, "Invalid credentials! Please try again.")
            auth_result[] = false
            # Clear the password field but leave the username
            set_gtk_property!(password_entry, :text, "")
        end
    end
    
    # Make the dialog close when the user hits Enter in the password field
    signal_connect(password_entry, "activate") do widget
        signal_emit(login_button, "clicked", Nothing)
    end
    
    # Prevent window close
    signal_connect(win, "delete-event") do widget, event
        set_gtk_property!(result_label, :label, "Authentication is required to proceed!")
        # Return true to prevent the window from closing
        return true
    end
    
    # Show all components
    showall(win)
    
    return win, auth_result
end

function authenticate()
    win, auth_result = create_auth_dialog()
    
    signal_connect(win, "destroy") do widget
        Gtk.quit()
    end
    
    Gtk.gtk_main()
    
    return auth_result[]
end

# Main program
function main()
    println("Starting authentication test...")
    
    # Show the authentication dialog and get the result
    result = authenticate()
    
    println("Authentication result: ", result)
    
    return result
end

# Run the program
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end