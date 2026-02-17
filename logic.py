class ComplianceController:
    """
    Manages the business logic for nationality mapping and 
    data visibility restrictions.
    """
    
    @staticmethod
    def derive_nationality(race_data):
        """
        Maps the AI-detected ethnicity to the project's specific
        nationality categories.
        """
        # Normalize input to avoid case-sensitivity issues
        detected_race = race_data.get('dominant_race', 'unknown').lower()
        
        # Custom Logic Mapping
        if detected_race == "indian":
            return "Indian"
        
        elif detected_race in ["black", "latino black"]:
            return "African"
        
        elif detected_race in ["white", "latino hispanic", "middle eastern"]:
            return "USA"  # Mapping these groups to USA as per requirement
        
        else:
            return "Other" # Asian, etc.

    @staticmethod
    def get_visibility_rules(nationality):
        """
        Returns a dictionary of booleans dictating what data to show.
        """
        # Default: Show nothing but Emotion
        rules = {
            "show_age": False,
            "show_dress": False,
            "show_emotion": True
        }

        if nationality == "Indian":
            rules.update({"show_age": True, "show_dress": True})
            
        elif nationality == "USA":
            rules.update({"show_age": True, "show_dress": False})
            
        elif nationality == "African":
            rules.update({"show_age": False, "show_dress": True})
            
        elif nationality == "Other":
            rules.update({"show_age": False, "show_dress": False})
            
        return rules