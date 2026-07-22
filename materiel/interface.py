"""Interface commune que doivent implémenter tous les "backends" pilotant le bras.

Le but : le reste du code (vision, logique de saisie) ne parle qu'à cette
interface, sans savoir s'il pilote la simulation MuJoCo ou les vrais servos
Feetech. On peut ainsi tester en simulation puis brancher le matériel réel
sans rien changer au code appelant.
"""

from abc import ABC, abstractmethod


class Bras(ABC):
    """Interface abstraite d'un bras robotisé à 5 degrés de liberté + pince."""

    @abstractmethod
    def aller_a(self, x: float, y: float, z: float) -> bool:
        """Déplace le bout de pince vers la position (x, y, z) en mètres.

        Doit utiliser cinematique.securite.ik_est_valide en interne et refuser
        (retourner False) tout mouvement non sûr, plutôt que de l'exécuter.
        Retourne True si le mouvement a été exécuté, False sinon.
        """
        raise NotImplementedError

    @abstractmethod
    def lire_positions(self):
        """Retourne les angles actuels des articulations (rad), dans l'ordre
        [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]."""
        raise NotImplementedError

    @abstractmethod
    def ouvrir_pince(self) -> None:
        """Ouvre complètement la pince (mâchoire mobile, joint 'gripper')."""
        raise NotImplementedError

    @abstractmethod
    def fermer_pince(self) -> None:
        """Ferme complètement la pince."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Arrête immédiatement tout mouvement en cours (arrêt d'urgence)."""
        raise NotImplementedError
