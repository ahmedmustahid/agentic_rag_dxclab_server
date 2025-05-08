import unicodedata

from dotenv import load_dotenv

load_dotenv()


class SearchError(Exception):
    """Exception thrown when an unknown error occurs during search processing"""

    pass


class SearchAnswerEngine:
    """
    SearchAnswerEngine
    Search and generate answers based on the results of the search
    """

    def __init__(self):
        """
        Constructor of SearchAnswerEngine class
        """
        pass

    def truncate_text(self, text: str, max_length: int = 10000) -> str:
        """
        The specified text is truncated if the total number of characters exceeds the specified maximum length, counting full-width characters as 2 and half-width characters as 1.

        Args:
          text (str): The string to be processed.
          max_length (int, optional): The maximum number of characters allowed. Default is 1500.

        Returns:
          str: The string truncated to fit within the specified maximum length.
        """

        def char_width(char):
            """
            Calculates the width of a character. If the character is full-width ('F', 'W', 'A'), use 2; otherwise use 1.

            Args:
              char (str): One character to calculate the width.

            Returns:
              int: Character width. If the character is full-width, use 2; otherwise use 1.
            """
            return 2 if unicodedata.east_asian_width(char) in "FWA" else 1

        current_length = 0
        result = []
        for char in text:
            current_length += char_width(char)
            if current_length > max_length:
                break
            result.append(char)
        return "".join(result)

    def repair_enc_univ(self, text) -> str:
        """
        Repair encoding universal
        Fix garbled characters in Tavily search results
        Attempts multiple encoding conversions and returns the most Japanese-like results
        No specific mapping table is used, just a general purpose

        Args:
          text (str): input text.

        Returns:
          text: Fixed text.
        """
        candidates = []
        original = text
        # Possible intermediate encodings
        intermediate_encodings = ["latin1", "iso-8859-1"]
        # Possible original encodings
        target_encodings = ["utf-8", "shift_jis", "euc-jp", "cp932"]
        # Try different encoding conversion paths
        for inter_enc in intermediate_encodings:
            try:
                # Converts the current string to a byte sequence using an intermediate encoding.
                byte_data = text.encode(inter_enc, errors="replace")
                # Attempts to decode various encodings
                for target_enc in target_encodings:
                    try:
                        decoded = byte_data.decode(target_enc, errors="replace")
                        candidates.append(decoded)
                    except Exception:
                        continue
                # Also try double encoding/decoding
                for target_enc in target_encodings:
                    try:
                        # Decode once, then encode/decode again
                        intermediate = byte_data.decode(target_enc, errors="replace")
                        for second_enc in target_encodings:
                            try:
                                re_encoded = intermediate.encode(
                                    second_enc, errors="replace"
                                )
                                final = re_encoded.decode("utf-8", errors="replace")
                                candidates.append(final)
                            except Exception:
                                continue
                    except Exception:
                        continue
            except Exception:
                continue

        # Evaluation of results (scoring Japanese-style)
        best_score = -1
        best_text = original
        for candidate in candidates:
            # The fewer replacement characters the better
            replacement_chars = candidate.count("�") + candidate.count("?")
            # The more Japanese characters the better
            jp_chars = sum(1 for c in candidate if ord(c) > 0x3000 and ord(c) < 0x30FF)
            # # Score calculation (the more Japanese characters and the fewer replacement characters, the higher the score)
            score = jp_chars * 2 - replacement_chars * 3
            if score > best_score:
                best_score = score
                best_text = candidate
        # If it is not an improvement over the original, return the original text.
        if best_score <= 0:
            return original
        return best_text
