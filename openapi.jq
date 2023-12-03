.components.schemas |= with_entries(
    .value |= 
        if .properties.content.type == "string" then
            .properties.content.format = "password"
        elif .properties.content.oneOf then
            .properties.content.oneOf |= map(
                if .type == "string" then
                    .format = "password"
                else
                    .
                end
            )
        else
            .
        end
)